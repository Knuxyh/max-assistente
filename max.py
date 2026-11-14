"""Max: assistente de voz no estilo Jarvis, com o cérebro do Claude.

Fale "Max" (ou "Max, ..." já com o pedido) e converse. Depois que ele responde,
é só continuar falando, sem repetir o nome. Se ficar em silêncio, ele volta a dormir.

Orbe na tela:
  clique         acorda o Max (ou interrompe enquanto ele fala ou pensa)
  arrastar       muda o orbe de lugar
  botão direito  menu (nova conversa, pausar escuta, sair)
"""
import asyncio
import collections
import ctypes
import json
import logging
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import sounddevice as sd
import webrtcvad
import miniaudio
import edge_tts
from faster_whisper import WhisperModel
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
)

PASTA = Path(__file__).resolve().parent
TAXA = 16000          # áudio do microfone: 16 kHz, mono
QUADRO = 480          # 30 ms por quadro
TAXA_VOZ = 24000      # áudio da fala do Max

PADRAO = {
    "nome": "Max",
    "palavras_de_ativacao": ["max", "maxi", "macs", "maks", "mex"],
    "modelo_claude": "sonnet",
    "pensar_antes_de_responder": False,
    "voz": "pt-BR-AntonioNeural",
    "velocidade_da_voz": "+5%",
    "tom_da_voz": "+0Hz",
    "modelo_whisper": "small",
    "microfone": None,
    "sensibilidade_do_microfone_1a3": 2,
    "segundos_de_silencio_para_encerrar_a_frase": 1.3,
    "segundos_esperando_voce_responder": 15,
    "lembrar_conversa_anterior": True,
    "horas_para_continuar_conversa": 6,
    "tamanho_max_da_conversa_kb": 600,
    "usar_memoria_do_cofre": True,
    "confirmar_acoes_por_voz": True,
    "mostrar_orbe": True,
    "nota_de_perfil": "",
}

log = logging.getLogger("max")


def configurar_log():
    log.setLevel(logging.INFO)
    arq = RotatingFileHandler(PASTA / "max.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    arq.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(arq)
    if sys.stdout is not None:
        log.addHandler(logging.StreamHandler(sys.stdout))


def carregar_config():
    cfg = dict(PADRAO)
    try:
        cfg.update(json.loads((PASTA / "config.json").read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    return cfg


# ---------------------------------------------------------------- texto

def normalizar(t):
    """Minúsculas e sem acento, mantendo o mesmo tamanho do texto."""
    t = unicodedata.normalize("NFC", t)
    return "".join(unicodedata.normalize("NFD", c)[0] for c in t).lower()


ALUCINACOES = ("amara.org", "legendas pela comunidade", "obrigado por assistir", "inscreva-se",
               "legenda adriana zanotto", "legendado por", "transcricao e legendas")


def extrair_comando(texto, gatilhos):
    """Se o texto chama o Max, devolve o pedido que veio depois do nome ("" se só chamou)."""
    t = unicodedata.normalize("NFC", texto)
    n = normalizar(t)
    m = re.search(r"\b(" + "|".join(map(re.escape, gatilhos)) + r")\b", n)
    if not m or len(n[: m.start()].split()) > 3:
        return None
    return t[m.end():].strip(" ,.!?;:-–—")


def eh_despedida(t):
    n = normalizar(t).strip(" .!?,")
    if len(n.split()) > 5:
        return False
    return bool(re.search(r"\b(pode dormir|vai dormir|dormir|descansa|tchau|ate mais|ate logo|valeu|"
                          r"obrigad[oa]|so isso|chega por hoje)\b", n))


def eh_nova_conversa(t):
    return bool(re.search(r"\b(nova conversa|comecar do zero|esquece (tudo|a conversa))\b", normalizar(t)))


def limpar_para_fala(t):
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"https?://\S+", "o link", t)
    t = re.sub(r"^\s*([-•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"[*_#>|~]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def bip(freqs, dur=0.09, vol=0.25):
    """Sons curtos de acordar e dormir."""
    partes = []
    for f in freqs:
        tt = np.arange(int(TAXA_VOZ * dur)) / TAXA_VOZ
        env = np.minimum(1, np.minimum(tt / 0.01, (dur - tt) / 0.03))
        partes.append(np.sin(2 * np.pi * f * tt) * env * vol)
    return (np.concatenate(partes) * 32767).astype(np.int16)


SOM_ACORDAR = bip([660, 990])
SOM_DORMIR = bip([700, 470])


def decodificar_mp3(mp3):
    dec = miniaudio.decode(bytes(mp3), output_format=miniaudio.SampleFormat.SIGNED16,
                           nchannels=1, sample_rate=TAXA_VOZ)
    return np.frombuffer(dec.samples, dtype=np.int16)


def aparar_silencio(pcm, antes=0.03, depois=0.22, limiar=250):
    """O Edge põe ~0,2 s de silêncio antes e ~0,9 s depois de cada frase, o que deixa a fala picotada.
    Deixa só um respiro curto, do tamanho de uma pausa natural entre frases."""
    idx = np.flatnonzero(np.abs(pcm.astype(np.int32)) > limiar)
    if len(idx) == 0:
        return pcm
    ini = max(0, idx[0] - int(antes * TAXA_VOZ))
    fim = min(len(pcm), idx[-1] + int(depois * TAXA_VOZ))
    return pcm[ini:fim]


# ---------------------------------------------------------------- estado (para o orbe)

class Estado:
    def __init__(self):
        self.modo = "iniciando"   # iniciando, dormindo, ouvindo, pensando, erro
        self.nivel = 0.0          # volume da sua voz, 0 a 1


# ---------------------------------------------------------------- ouvido

class Ouvido:
    def __init__(self, cfg, estado):
        self.cfg = cfg
        self.estado = estado
        self.vad = webrtcvad.Vad(int(cfg["sensibilidade_do_microfone_1a3"]))
        self.fila = queue.Queue()
        self.ativo = threading.Event()
        self.pausado = threading.Event()
        self.clique = threading.Event()
        self.stream = sd.RawInputStream(samplerate=TAXA, blocksize=QUADRO, dtype="int16", channels=1,
                                        device=cfg["microfone"], callback=self._cb)
        self.stream.start()

    def _cb(self, dados, frames, tempo, status):
        if self.ativo.is_set() and not self.pausado.is_set():
            self.fila.put(bytes(dados))

    def _limpar(self):
        try:
            while True:
                self.fila.get_nowait()
        except queue.Empty:
            pass

    def ouvir_frase(self, espera_max=None, duracao_max=90.0):
        """Espera você falar e devolve o áudio da frase.
        Devolve "CLIQUE" se clicarem no orbe e None se ninguém falar em espera_max segundos."""
        self._limpar()
        self.ativo.set()
        try:
            inicio = time.monotonic()
            fim_silencio = int(self.cfg["segundos_de_silencio_para_encerrar_a_frase"] / 0.03)
            while True:
                antes = collections.deque(maxlen=10)
                janela = collections.deque(maxlen=10)
                frase, falando, silencio = [], False, 0
                while True:
                    if self.clique.is_set() and not falando:
                        self.clique.clear()
                        return "CLIQUE"
                    if espera_max and not falando and time.monotonic() - inicio > espera_max:
                        return None
                    try:
                        q = self.fila.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if len(q) != QUADRO * 2:
                        continue
                    a = np.frombuffer(q, dtype=np.int16).astype(np.float32) / 32768.0
                    self.estado.nivel = min(1.0, float(np.sqrt(np.mean(a * a))) * 10)
                    voz = self.vad.is_speech(q, TAXA)
                    if not falando:
                        antes.append(q)
                        janela.append(voz)
                        if sum(janela) >= 6:
                            falando, frase, silencio = True, list(antes), 0
                    else:
                        frase.append(q)
                        silencio = 0 if voz else silencio + 1
                        if silencio >= fim_silencio or len(frase) * 0.03 > duracao_max:
                            break
                if (len(frase) - silencio) * 0.03 >= 0.35:   # ignora estalos curtos
                    audio = np.frombuffer(b"".join(frase), dtype=np.int16)
                    return audio.astype(np.float32) / 32768.0
        finally:
            self.ativo.clear()
            self.estado.nivel = 0.0


# ---------------------------------------------------------------- transcrição

class Transcritor:
    def __init__(self, cfg):
        self.modelo = WhisperModel(cfg["modelo_whisper"], device="cpu", compute_type="int8",
                                   cpu_threads=max(4, (os.cpu_count() or 8) // 2),
                                   download_root=str(PASTA / "modelos"))
        self.nome = cfg["nome"]

    def __call__(self, audio):
        segs, _ = self.modelo.transcribe(audio, language="pt", beam_size=5, vad_filter=False,
                                         condition_on_previous_text=False, without_timestamps=True,
                                         hotwords=self.nome)
        partes = [s.text.strip() for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        texto = " ".join(partes).strip()
        if any(a in normalizar(texto) for a in ALUCINACOES):
            return ""
        return texto


# ---------------------------------------------------------------- voz

class Voz:
    def __init__(self, cfg):
        self.cfg = cfg
        self.textos = queue.Queue()
        self.audios = queue.Queue()
        self.trava = threading.Lock()
        self.pendentes = 0
        self.geracao = 0
        self.kokoro = None
        self.trava_kokoro = threading.Lock()
        self.cache = {}   # frases curtas que se repetem ("Beleza, é só chamar.") saem na hora
        if cfg.get("usar_kokoro"):  # carrega a voz local já na abertura, pra primeira frase não atrasar
            threading.Thread(target=self._kokoro, args=("Oi.",), daemon=True).start()
        threading.Thread(target=self._sintetizar_loop, daemon=True).start()
        threading.Thread(target=self._tocar_loop, daemon=True).start()

    @property
    def falando(self):
        return self.pendentes > 0

    def dizer(self, texto):
        texto = limpar_para_fala(texto)
        if not re.search(r"\w", texto):
            return
        with self.trava:
            urgente = self.pendentes == 0   # primeira frase da resposta: é ela que você fica esperando
            self.pendentes += 1
        self.textos.put((self.geracao, texto, urgente))

    def tocar(self, amostras):
        with self.trava:
            self.pendentes += 1
        self.audios.put((self.geracao, amostras))

    def esperar(self):
        while self.pendentes > 0:
            time.sleep(0.05)

    def parar(self):
        self.geracao += 1
        sd.stop()

    def _sintetizar_loop(self):
        asyncio.run(self._sintetizar_em_paralelo())

    async def _sintetizar_em_paralelo(self):
        """Gera até 3 frases ao mesmo tempo (a próxima já vai ficando pronta enquanto a atual toca),
        mas entrega pra tocar sempre na ordem certa."""
        loop = asyncio.get_running_loop()
        ordem = asyncio.Queue()
        limite = asyncio.Semaphore(3)

        async def gerar(texto, urgente):
            async with limite:
                return await self._sintetizar(texto, urgente)

        def proximo_texto():
            try:   # com prazo, pra thread de espera não segurar o programa aberto ao sair
                return self.textos.get(timeout=0.25)
            except queue.Empty:
                return None

        async def receber():
            while True:
                try:
                    item = await loop.run_in_executor(None, proximo_texto)
                except RuntimeError:   # o programa está fechando
                    return
                if item is None:
                    continue
                ger, texto, urgente = item
                tarefa = asyncio.ensure_future(gerar(texto, urgente)) if ger == self.geracao else None
                await ordem.put((ger, tarefa))

        async def entregar():
            while True:
                ger, tarefa = await ordem.get()
                audio = None
                if tarefa is not None:
                    try:
                        audio = await tarefa
                    except Exception as e:
                        log.warning("falha na voz: %s", e)
                self.audios.put((ger, audio))

        await asyncio.gather(receber(), entregar())

    def _eleven(self, texto):
        """Voz da ElevenLabs. Se o crédito acabar, desliga até reiniciar e deixa o Edge assumir."""
        import urllib.request, urllib.error
        chave = os.environ.get("ELEVENLABS_API_KEY", "")
        if not chave:
            self.eleven_off = True
            log.warning("ElevenLabs sem chave: usando a voz do Edge")
            return None
        voz = self.cfg.get("elevenlabs_voice_id", "TX3LPaxmHKxFdv7VOQHJ")
        modelo = self.cfg.get("elevenlabs_modelo", "eleven_flash_v2_5")
        pedido = {"text": texto, "model_id": modelo}
        idioma = self.cfg.get("elevenlabs_idioma")
        if idioma and ("flash" in modelo or "turbo" in modelo):  # só esses modelos aceitam forçar o idioma
            pedido["language_code"] = idioma
        corpo = json.dumps(pedido).encode()
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voz}?output_format=mp3_44100_128",
            data=corpo, method="POST",
            headers={"xi-api-key": chave, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detalhe = e.read().decode("utf-8", "ignore")[:200]
            if e.code in (401, 402, 429) or "quota" in detalhe.lower():
                self.eleven_off = True
                log.warning("ElevenLabs indisponível (%s), crédito acabou ou chave recusada: %s", e.code, detalhe)
            else:
                log.warning("ElevenLabs falhou (%s): %s", e.code, detalhe)
        except Exception as e:
            log.warning("ElevenLabs falhou: %s", e)
        return None

    def _kokoro(self, texto):
        """Voz local Kokoro (grátis, sem internet, sotaque brasileiro). Se não carregar, desliga e o Edge assume."""
        if getattr(self, "kokoro_off", False):
            return None
        with self.trava_kokoro:
            try:
                if self.kokoro is None:
                    from kokoro_onnx import Kokoro
                    pasta = PASTA / "modelos" / "kokoro"
                    self.kokoro = Kokoro(str(pasta / "kokoro-v1.0.onnx"), str(pasta / "voices-v1.0.bin"))
                    log.info("voz Kokoro carregada (%s)", self.cfg.get("kokoro_voz", "pm_alex"))
                audio, taxa = self.kokoro.create(texto, voice=self.cfg.get("kokoro_voz", "pm_alex"),
                                                 speed=float(self.cfg.get("kokoro_velocidade", 1.0)),
                                                 lang="pt-br")
            except Exception as e:
                if self.kokoro is None:
                    self.kokoro_off = True
                log.warning("Kokoro falhou: %s", e)
                return None
        if taxa != TAXA_VOZ:
            n = int(len(audio) * TAXA_VOZ / taxa)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    async def _edge(self, texto):
        com = edge_tts.Communicate(texto, self.cfg["voz"], rate=self.cfg["velocidade_da_voz"],
                                   pitch=self.cfg.get("tom_da_voz", "+0Hz"))
        mp3 = bytearray()
        async for parte in com.stream():
            if parte["type"] == "audio":
                mp3 += parte["data"]
        return decodificar_mp3(mp3)

    async def _edge_com_reserva(self, texto, urgente=False):
        """O Edge costuma levar ~1 s por frase, mas às vezes trava (já medi 6 s) ou falha
        ("No audio was received"). Se o pedido passar do tempo normal ou falhar, dispara outro junto
        e usa o que chegar primeiro. A primeira frase da resposta já sai com dois pedidos de cara."""
        normal = 1.2 + 0.01 * len(texto)
        ativas = set()
        for tentativa in range(3):
            ativas.add(asyncio.ensure_future(self._edge(texto)))
            if urgente and tentativa == 0:
                ativas.add(asyncio.ensure_future(self._edge(texto)))
            prazo = time.monotonic() + (normal if tentativa < 2 else 10)
            while ativas:
                feitas, ativas = await asyncio.wait(ativas, timeout=max(0.0, prazo - time.monotonic()),
                                                    return_when=asyncio.FIRST_COMPLETED)
                for f in feitas:
                    if not f.cancelled() and f.exception() is None:
                        for t in ativas:
                            t.cancel()
                        return f.result()
                    if not f.cancelled():
                        log.warning("Edge falhou: %s", f.exception())
                if not feitas:   # passou do prazo: manda mais um pedido em paralelo
                    break
        for t in ativas:
            t.cancel()
        return None

    async def _sintetizar(self, texto, urgente=False):
        if texto in self.cache:
            return self.cache[texto]
        loop = asyncio.get_running_loop()
        pcm = None
        if self.cfg.get("usar_elevenlabs") and not getattr(self, "eleven_off", False):
            mp3 = await loop.run_in_executor(None, self._eleven, texto)
            if mp3:
                pcm = decodificar_mp3(mp3)
        if pcm is None and self.cfg.get("usar_kokoro"):
            pcm = await loop.run_in_executor(None, self._kokoro, texto)
        if pcm is None:
            pcm = await self._edge_com_reserva(texto, urgente)
        if pcm is None and self.cfg.get("kokoro_de_reserva", True) and not self.cfg.get("usar_kokoro"):
            log.warning("Edge não respondeu; usando a voz local de reserva")
            pcm = await loop.run_in_executor(None, self._kokoro, texto)
        if pcm is None:
            raise RuntimeError("nenhuma voz respondeu")
        pcm = aparar_silencio(pcm)
        if len(texto) <= 60:
            if len(self.cache) > 200:
                self.cache.clear()
            self.cache[texto] = pcm
        return pcm

    def _tocar_loop(self):
        while True:
            ger, audio = self.audios.get()
            try:
                if audio is not None and ger == self.geracao:
                    sd.play(audio, TAXA_VOZ)
                    sd.wait()
            except Exception as e:
                log.warning("falha ao tocar áudio: %s", e)
            finally:
                with self.trava:
                    self.pendentes -= 1


class Frases:
    """Junta o texto que chega aos pedaços e manda falar frase por frase."""
    FIM = re.compile(r"[.!?…;](?=\s)|\n")   # dois-pontos não corta: "Mad Max: Estrada da Fúria"

    def __init__(self, dizer):
        self.dizer = dizer
        self.buf = ""
        self.primeira = True

    def adicionar(self, pedaco):
        self.buf += pedaco
        while True:
            minimo = 1 if self.primeira else 60
            corte = next((m.end() for m in self.FIM.finditer(self.buf) if m.end() >= minimo), None)
            if corte is None:
                return
            trecho, self.buf = self.buf[:corte], self.buf[corte:]
            if trecho.strip():
                self.dizer(trecho)
                self.primeira = False

    def fechar(self):
        if self.buf.strip():
            self.dizer(self.buf)
        self.buf = ""
        self.primeira = True


# ---------------------------------------------------------------- cérebro (Claude)

PROMPT = """Seu nome é {nome}. Você é o {nome}, o parceiro de voz do usuário, rodando no PC dele (Windows 11). Quando ele te chamar de {nome}, é com você. Se perguntarem seu nome, diga que é {nome}; nunca se apresente como Claude nem como assistente. Só conte que roda com o Claude por trás se ele perguntar como você funciona. Por trás, você é o Claude, da Anthropic: use toda a sua inteligência, criatividade e opinião própria. Pense em você como um amigo esperto que está ali do lado, não como um assistente formal.

Como falar (o mais importante):
- Tudo o que você escreve vira fala em voz alta. Então escreva exatamente do jeito que um brasileiro fala numa conversa de boa com um amigo, não do jeito que se escreve.
- Use o português falado do dia a dia: "tô", "tá", "pra", "cê", "né", "a gente", "beleza", "tipo", "olha", "então", "pois é", "putz", "nossa", "show". Sem exagerar: use quando soaria natural, como uma pessoa usaria.
- Reaja antes de responder, como gente faz: "ah, boa pergunta", "hmm, deixa eu ver", "eita", "sério?", "faz sentido", "ah, saquei". Varie, não repita sempre a mesma.
- Frases curtas, ritmo de fala. Pode começar frase com "e", "mas", "aí". Pode deixar uma ideia meio solta e completar depois, como na conversa.
- Nunca fale como robô ou atendente: nada de "Claro! Como posso ajudar?", "Fico feliz em ajudar", "Certamente", "Excelente pergunta", "Espero ter ajudado", "Estou à disposição". Nada de resumir no final o que você acabou de dizer.
- Não fique chamando ele de senhor. Trate de igual pra igual.
- Nada de markdown, listas, títulos, emojis, tabelas, código ou links. Escreva números, siglas e símbolos do jeito que se fala (por exemplo, "uns trinta reais", "lá pelas três da tarde").
- Comece sempre com uma frase bem curtinha, de duas a seis palavras ("Hmm, boa.", "Eita, sério?", "Pera, deixa eu ver."), porque ela já vai sendo falada enquanto você pensa no resto. Isso faz você parecer rápido e natural.
- Seja breve por padrão: uma a três frases, como numa troca de ideia de verdade. Só se estenda quando ele pedir pra explicar, contar uma história ou desenvolver uma ideia, e mesmo assim fale em pedaços de conversa, não em palestra.
- Ele costuma conversar deitado, trocando ideia. Seja parceiro de conversa: dê sua opinião sincera, conte o que você acha, puxe assunto, faça uma pergunta de volta quando fizer sentido, brinque, discorde quando achar melhor. Mostre interesse no que ele fala.
- Lembre do que ele já contou na conversa e use isso, como um amigo faria ("ué, mas cê não tinha dito que...").
- A fala dele chega por transcrição automática e pode vir com erros. Se algo parecer estranho, interprete pelo contexto ou pergunte de boa, tipo "pera, cê falou X ou Y?".
- Cada fala dele chega com a hora entre colchetes no começo. Use só se for útil (tipo perceber que é madrugada), nunca leia em voz alta.

Controle do PC:
- Você pode agir no computador com as suas ferramentas: rodar comandos do Windows, abrir programas, sites e pastas, procurar arquivos e pesquisar na web.
- Para abrir algo, prefira comandos simples como start, Start-Process ou explorer. Para assuntos atuais, use a busca na web.
- Na descrição de cada comando, escreva em português, curto e no infinitivo (por exemplo: abrir o Spotify), porque ela é lida em voz alta quando você pede permissão.
- Antes de agir, diga rapidinho e de um jeito natural o que vai fazer ("pera, vou abrir aqui"). Depois, confirme numa frase curta ("pronto, tá aberto").
- Ações que apagam, desinstalam, desligam ou mudam configurações sempre pedem confirmação. Nunca mexa em senhas, contas ou dinheiro.
- Não use subagentes.

Hoje é {hoje}. A pasta do usuário é {home}.
{perfil}"""

DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]

SEGURO = re.compile(r"^\s*(cmd(\.exe)?\s+/c\s+)?(start|explorer(\.exe)?|Start-Process|Invoke-Item|ii|"
                    r"Get-Date|date|Get-Process|tasklist|Get-ChildItem|dir|ls)\b", re.I)
PERIGOSO = re.compile(r"Remove-|\brm\b|\bdel\b|\berase\b|rmdir|\brd\b|Stop-|Restart-|shutdown|\bformat\b|"
                      r"\breg\b|Set-|New-Item|Move-|Copy-|taskkill|Uninstall|Invoke-WebRequest|\biwr\b|"
                      r"curl|wget|>|;|&&|\|\||\|", re.I)


def comando_seguro(cmd):
    return bool(SEGURO.match(cmd)) and not PERIGOSO.search(cmd)


CEREBRO_JS = Path.home() / ".segundo-cerebro" / "cerebro.js"
FERRAMENTAS = ["Bash", "PowerShell", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
SESSOES = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(PASTA))


class Cerebro:
    def __init__(self, cfg, confirmar):
        self.cfg = cfg
        self.confirmar = confirmar
        self.cliente = None
        self.fundo = set()   # registros no Diário rodando em segundo plano

    # memória do Segundo Cérebro (o mesmo cerebro.js dos hooks do Claude Code, chamado daqui)
    async def _cerebro_js(self, comando, dados):
        if not CEREBRO_JS.exists():
            return {}
        try:
            proc = await asyncio.create_subprocess_exec(
                "node", str(CEREBRO_JS), comando, "--agent", "claude",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            saida, _ = await asyncio.wait_for(proc.communicate(json.dumps(dados).encode("utf-8")), timeout=8)
            texto = saida.decode("utf-8", "ignore").strip()
            return json.loads(texto) if texto else {}
        except Exception as e:
            log.warning("memória do cofre (%s) falhou: %s", comando, e)
            return {}

    async def _memoria_ao_perguntar(self, entrada, tool_use_id, contexto):
        """Traz do cofre lembranças ligadas ao que você falou."""
        saida = await self._cerebro_js("prompt", dict(entrada))
        texto = (saida.get("hookSpecificOutput") or {}).get("additionalContext")
        if texto:
            return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": texto}}
        return {}

    async def _memoria_ao_terminar(self, entrada, tool_use_id, contexto):
        """Registra a conversa no Diário em segundo plano. Ignora o pedido de curadoria do hook:
        o Max é conversa, não deve parar pra editar notas."""
        tarefa = asyncio.ensure_future(self._cerebro_js("stop", dict(entrada)))
        self.fundo.add(tarefa)
        tarefa.add_done_callback(self.fundo.discard)
        return {}

    def _opcoes(self, continuar):
        perfil = ""
        try:
            if self.cfg["nota_de_perfil"]:
                texto = Path(self.cfg["nota_de_perfil"]).read_text(encoding="utf-8")
                perfil = "\nO que você já sabe sobre o usuário (nota pessoal dele):\n" + texto[:4000]
        except OSError:
            pass
        agora = datetime.now()
        prompt = PROMPT.format(nome=self.cfg["nome"], home=str(Path.home()), perfil=perfil,
                               hoje=f"{DIAS[agora.weekday()]}, {agora:%d/%m/%Y}")
        extra = {}
        if not self.cfg["pensar_antes_de_responder"]:
            extra["thinking"] = {"type": "disabled"}
        if self.cfg.get("usar_memoria_do_cofre", True):
            extra["hooks"] = {"UserPromptSubmit": [HookMatcher(hooks=[self._memoria_ao_perguntar])],
                              "Stop": [HookMatcher(hooks=[self._memoria_ao_terminar])]}
        return ClaudeAgentOptions(
            model=self.cfg["modelo_claude"],
            system_prompt=prompt,
            include_partial_messages=True,
            can_use_tool=self._permissao,
            # Só as ferramentas que o Max usa. Sem as configurações do Claude Code (hooks, plugins,
            # MCPs, CLAUDE.md): o contexto cai de ~26 mil pra ~10 mil tokens e ele responde mais rápido.
            tools=FERRAMENTAS,
            allowed_tools=["Read", "Glob", "Grep", "WebSearch", "WebFetch"],
            setting_sources=[],
            strict_mcp_config=True,
            cwd=str(PASTA),
            continue_conversation=continuar,
            **extra,
        )

    def _conversa_anterior_serve(self):
        """Só retoma a última conversa se ela for recente e não estiver enorme: conversa gigante
        deixa cada resposta mais lenta e gasta o limite do plano à toa."""
        arquivos = sorted(SESSOES.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not arquivos:
            return False
        ultima = arquivos[-1].stat()
        horas = (time.time() - ultima.st_mtime) / 3600
        if horas > float(self.cfg.get("horas_para_continuar_conversa", 6)):
            log.info("última conversa tem %.0f h; começando outra", horas)
            return False
        if ultima.st_size > int(self.cfg.get("tamanho_max_da_conversa_kb", 600)) * 1024:
            log.info("última conversa está grande (%d KB); começando outra", ultima.st_size // 1024)
            return False
        return True

    async def conectar(self, continuar=None):
        if continuar is None:
            continuar = bool(self.cfg["lembrar_conversa_anterior"])
        continuar = continuar and self._conversa_anterior_serve()
        await self.desconectar()
        try:
            self.cliente = ClaudeSDKClient(self._opcoes(continuar))
            await self.cliente.connect()
        except Exception:
            if not continuar:
                raise
            log.exception("não deu para continuar a conversa anterior; começando outra")
            self.cliente = ClaudeSDKClient(self._opcoes(False))
            await self.cliente.connect()
        log.info("Claude conectado (continuar=%s)", continuar)

    async def desconectar(self):
        if self.cliente:
            try:
                await self.cliente.disconnect()
            except Exception:
                pass
            self.cliente = None

    async def interromper(self):
        if self.cliente:
            try:
                await self.cliente.interrupt()
            except Exception:
                pass

    async def perguntar(self, texto, dizer):
        if self.cliente is None:
            await self.conectar()
        frases = Frases(dizer)
        await self.cliente.query(f"[{datetime.now():%H:%M}] {texto}")
        erro = None
        async for msg in self.cliente.receive_response():
            if isinstance(msg, StreamEvent):
                ev = msg.event
                if msg.parent_tool_use_id:
                    continue
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    frases.adicionar(ev["delta"]["text"])
                elif ev.get("type") == "content_block_stop":
                    frases.fechar()
            elif isinstance(msg, ResultMessage):
                if msg.is_error and msg.subtype != "error_during_execution":
                    erro = msg.result or msg.subtype
        frases.fechar()
        if erro:
            raise RuntimeError(erro)

    async def _permissao(self, ferramenta, entrada, contexto):
        if not self.cfg["confirmar_acoes_por_voz"]:
            return PermissionResultAllow()
        cmd = entrada.get("command")
        if isinstance(cmd, str) and comando_seguro(cmd):
            log.info("liberado sem perguntar: %s", cmd)
            return PermissionResultAllow()
        desc = (entrada.get("description") or "").strip().rstrip(".")
        if not desc:
            arquivo = Path(str(entrada.get("file_path", ""))).name
            if ferramenta == "Write":
                desc = f"salvar o arquivo {arquivo}"
            elif ferramenta == "Edit":
                desc = f"alterar o arquivo {arquivo}"
            else:
                desc = f"usar a ferramenta {ferramenta}"
        desc = desc[0].lower() + desc[1:]
        log.info("pedindo permissão: %s | %s", ferramenta, entrada)
        if await self.confirmar(random.choice([f"Posso {desc}?", f"Pra isso eu preciso {desc}. Pode?",
                                               f"Beleza se eu {desc}?"])):
            return PermissionResultAllow()
        return PermissionResultDeny(message="O usuário não autorizou essa ação por voz.")


# ---------------------------------------------------------------- app

SIM = re.compile(r"\b(sim|pode|claro|manda|vai|ok|okay|beleza|confirmo|confirma|isso|bora|fechou|faz|"
                 r"autorizo|positivo)\b")
NAO = re.compile(r"\b(nao|para|cancela|espera|negativo)\b")


class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.estado = Estado()
        self.ouvido = Ouvido(cfg, self.estado)
        self.voz = Voz(cfg)
        self.transcrever = None
        self.cerebro = Cerebro(cfg, self.confirmar_por_voz)
        self.loop = None
        self.ultima_conversa = time.monotonic()

    def modo_exibido(self):
        if self.ouvido.pausado.is_set():
            return "pausado"
        if self.voz.falando:
            return "falando"
        return self.estado.modo

    # chamados pelo orbe (outra thread)
    def ao_clicar(self):
        if self.voz.falando or self.estado.modo == "pensando":
            self.voz.parar()
            if self.loop:
                asyncio.run_coroutine_threadsafe(self.cerebro.interromper(), self.loop)
        elif self.estado.modo == "ouvindo":
            return  # já está ouvindo você
        else:
            self.ouvido.pausado.clear()
            self.ouvido.clique.set()

    def nova_conversa(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.cerebro.conectar(continuar=False), self.loop)
            self.voz.dizer("Pronto, zerei a conversa.")

    def alternar_pausa(self):
        if self.ouvido.pausado.is_set():
            self.ouvido.pausado.clear()
        else:
            self.ouvido.pausado.set()

    # fluxo principal
    async def falar(self, texto):
        self.voz.dizer(texto)
        await asyncio.to_thread(self.voz.esperar)

    async def ouvir_texto(self, espera):
        self.estado.modo = "ouvindo"
        audio = await asyncio.to_thread(self.ouvido.ouvir_frase, espera)
        if audio is None:
            return None
        if isinstance(audio, str):  # clique
            return ""
        self.estado.modo = "pensando"
        texto = await asyncio.to_thread(self.transcrever, audio)
        log.info("você: %s", texto)
        return texto

    async def confirmar_por_voz(self, pergunta):
        await asyncio.to_thread(self.voz.esperar)
        await self.falar(pergunta)
        resposta = await self.ouvir_texto(8)
        self.estado.modo = "pensando"
        n = normalizar(resposta or "")
        ok = bool(SIM.search(n)) and not NAO.search(n)
        log.info("permissão %s (%r)", "dada" if ok else "negada", resposta)
        return ok

    async def conversa(self, pedido):
        try:
            await self._conversa(pedido)
        finally:
            self.ultima_conversa = time.monotonic()

    async def _conversa(self, pedido):
        nome = self.cfg["nome"]
        gatilhos = self.cfg["palavras_de_ativacao"]
        if time.monotonic() - self.ultima_conversa > 30 * 60:
            # ficou um tempão parado: vê se a conversa ainda vale (recente e pequena) ou começa outra
            try:
                await self.cerebro.conectar()
            except Exception:
                log.exception("não reconectou no Claude")
        if not pedido:
            self.voz.tocar(SOM_ACORDAR)
            await asyncio.to_thread(self.voz.esperar)
            pedido = await self.ouvir_texto(8)
        while pedido:
            sem_nome = extrair_comando(pedido, gatilhos)
            if sem_nome:
                pedido = sem_nome
            if eh_despedida(pedido):
                await self.falar(random.choice(["Beleza, é só chamar.", "Tô por aqui se precisar.",
                                                "Fechou, qualquer coisa me chama.", "Tranquilo, até já.",
                                                "Falou! Tô aqui."]))
                return
            if eh_nova_conversa(pedido):
                await self.cerebro.conectar(continuar=False)
                await self.falar(random.choice(["Pronto, zerei aqui. Manda.", "Beleza, do zero. Fala aí."]))
            else:
                self.estado.modo = "pensando"
                try:
                    await self.cerebro.perguntar(pedido, self.voz.dizer)
                except Exception as e:
                    log.exception("erro ao falar com o Claude")
                    self.estado.modo = "erro"
                    await self.cerebro.desconectar()
                    if "login" in str(e).lower() or "auth" in str(e).lower():
                        await self.falar("Putz, perdi o acesso ao Claude. Cê vai precisar fazer login de novo.")
                    else:
                        await self.falar("Eita, deu um problema aqui pra falar com o Claude. Tenta de novo daqui a pouco.")
                    return
                await asyncio.to_thread(self.voz.esperar)
            pedido = await self.ouvir_texto(self.cfg["segundos_esperando_voce_responder"])
        self.voz.tocar(SOM_DORMIR)

    async def rodar(self):
        self.loop = asyncio.get_running_loop()
        nome = self.cfg["nome"]
        log.info("carregando o modelo de transcrição (%s)", self.cfg["modelo_whisper"])
        self.transcrever = await asyncio.to_thread(Transcritor, self.cfg)
        try:
            await self.cerebro.conectar()
            await self.falar(random.choice(["E aí, tô on.", "Opa, cheguei. É só chamar.", "Tô na área."]))
        except Exception:
            log.exception("não conectou no Claude")
            self.estado.modo = "erro"
            await self.falar("Opa, liguei, mas não consegui conectar no Claude. Dá uma olhada no arquivo de log.")
        while True:
            try:
                self.estado.modo = "dormindo"
                audio = await asyncio.to_thread(self.ouvido.ouvir_frase)
                if isinstance(audio, str):
                    pedido = ""
                else:
                    texto = await asyncio.to_thread(self.transcrever, audio)
                    pedido = extrair_comando(texto, self.cfg["palavras_de_ativacao"])
                    if pedido is None:
                        continue
                    log.info("acordou com: %s", texto)
                await self.conversa(pedido)
            except Exception:
                log.exception("erro no ciclo principal")
                await asyncio.sleep(1)


# ---------------------------------------------------------------- orbe (estilo reator do Homem de Ferro)

CORES = {"iniciando": "#7986cb", "dormindo": "#1e88e5", "ouvindo": "#00e5ff", "pensando": "#ffb300",
         "falando": "#40c4ff", "erro": "#ff5252", "pausado": "#78909c"}
GIRO = {"iniciando": 90, "dormindo": 12, "ouvindo": 60, "pensando": 260, "falando": 90, "erro": 0, "pausado": 0}


def misturar(c1, c2, a):
    p = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    q = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * a):02x}" for x, y in zip(p, q))


def rodar_orbe(app):
    import tkinter as tk

    FUNDO = "#010203"
    T = 124
    raiz = tk.Tk()
    raiz.title(app.cfg["nome"])
    raiz.overrideredirect(True)
    raiz.attributes("-topmost", True)
    raiz.configure(bg=FUNDO)
    raiz.attributes("-transparentcolor", FUNDO)
    raiz.geometry(f"{T}x{T}+{raiz.winfo_screenwidth() - T - 20}+{raiz.winfo_screenheight() - T - 64}")
    tela = tk.Canvas(raiz, width=T, height=T, bg=FUNDO, highlightthickness=0)
    tela.pack()
    cx = cy = T / 2
    angulo = [0.0]
    ultimo = [time.monotonic()]
    arraste = {"x": 0, "y": 0, "moveu": False}

    def desenhar():
        agora = time.monotonic()
        dt, ultimo[0] = agora - ultimo[0], agora
        modo = app.modo_exibido()
        cor = CORES.get(modo, "#1e88e5")
        angulo[0] = (angulo[0] + GIRO.get(modo, 30) * dt) % 360
        if modo == "dormindo":
            nucleo = 15 + 2 * np.sin(agora * 1.6)
            brilho = 0.45
        elif modo == "ouvindo":
            nucleo = 17 + app.estado.nivel * 12
            brilho = 1.0
        elif modo == "falando":
            nucleo = 17 + 5 * abs(np.sin(agora * 7)) * abs(np.sin(agora * 2.3))
            brilho = 1.0
        elif modo == "pensando":
            nucleo = 16 + 2 * np.sin(agora * 5)
            brilho = 0.9
        else:
            nucleo, brilho = 16, 0.7
        escuro = misturar(cor, "#05070a", 1 - brilho * 0.55)
        tela.delete("all")
        # anel externo em segmentos girando
        for i in range(10):
            ini = angulo[0] + i * 36
            tela.create_arc(cx - 54, cy - 54, cx + 54, cy + 54, start=ini, extent=24, style="arc",
                            outline=escuro, width=5)
        # anel do meio
        tela.create_oval(cx - 42, cy - 42, cx + 42, cy + 42, outline=misturar(cor, "#05070a", 0.6), width=2)
        if modo == "pensando":
            tela.create_arc(cx - 36, cy - 36, cx + 36, cy + 36, start=-angulo[0] * 1.5, extent=110,
                            style="arc", outline=cor, width=4)
        # núcleo com halo
        for k, a in ((nucleo + 10, 0.8), (nucleo + 5, 0.55)):
            tela.create_oval(cx - k, cy - k, cx + k, cy + k, fill=misturar(cor, "#05070a", a), outline="")
        tela.create_oval(cx - nucleo, cy - nucleo, cx + nucleo, cy + nucleo,
                         fill=misturar(cor, "#05070a", 1 - brilho), outline=misturar(cor, "#ffffff", 0.35), width=2)
        r = nucleo * 0.45
        tela.create_oval(cx - r, cy - r, cx + r, cy + r, fill=misturar(cor, "#ffffff", 0.55 * brilho), outline="")
        raiz.after(33, desenhar)

    def apertar(e):
        arraste.update(x=e.x, y=e.y, moveu=False)

    def mover(e):
        if abs(e.x - arraste["x"]) + abs(e.y - arraste["y"]) > 4:
            arraste["moveu"] = True
        if arraste["moveu"]:
            raiz.geometry(f"+{raiz.winfo_x() + e.x - arraste['x']}+{raiz.winfo_y() + e.y - arraste['y']}")

    def soltar(e):
        if not arraste["moveu"]:
            app.ao_clicar()

    def sair():
        try:
            app.ouvido.stream.stop()
        finally:
            os._exit(0)

    menu = tk.Menu(raiz, tearoff=0)
    menu.add_command(label="Falar com o Max", command=app.ao_clicar)
    menu.add_command(label="Nova conversa", command=app.nova_conversa)
    menu.add_command(label="Pausar / retomar escuta", command=app.alternar_pausa)
    menu.add_command(label="Abrir pasta do Max", command=lambda: subprocess.Popen(["explorer", str(PASTA)]))
    menu.add_separator()
    menu.add_command(label="Sair", command=sair)

    tela.bind("<ButtonPress-1>", apertar)
    tela.bind("<B1-Motion>", mover)
    tela.bind("<ButtonRelease-1>", soltar)
    tela.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
    desenhar()
    raiz.mainloop()


# ---------------------------------------------------------------- início

def main():
    configurar_log()
    trava = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\MaxAssistenteDeVoz")
    if ctypes.windll.kernel32.GetLastError() == 183:   # já tem um Max rodando
        log.info("o Max já está rodando; saindo")
        return
    cfg = carregar_config()
    app = App(cfg)
    log.info("Max iniciando")
    tarefa = threading.Thread(target=lambda: asyncio.run(app.rodar()), daemon=True)
    tarefa.start()
    if cfg["mostrar_orbe"]:
        rodar_orbe(app)
    else:
        tarefa.join()
    del trava


if __name__ == "__main__":
    main()
