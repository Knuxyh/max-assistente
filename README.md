# Max

Assistente de voz em português, no estilo Jarvis, com o cérebro do Claude (via `claude-agent-sdk`).
Você fala "Max" (ou "Max, ..." já com o pedido), ele ouve com o Whisper, responde com o Claude e fala com a voz do Edge.
Um orbe na tela mostra o estado: clique acorda ou interrompe, botão direito abre o menu.

## Requisitos

- Windows 11 e Python 3.12
- Microfone e internet (a voz do Edge e o Claude são online)
- Conta do Claude (plano Pro ou superior): o Max usa o login do Claude Code, sem chave de API
- Node.js, só se for usar a memória do cofre (`~\.segundo-cerebro\cerebro.js`)

## Instalação

```powershell
cd caminho\da\pasta\Max
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy config.exemplo.json config.json
```

Faça o login do Claude uma vez (abre um link no navegador):

```powershell
& ".venv\Lib\site-packages\claude_agent_sdk\_bundled\claude.exe" auth login --claudeai
```

O modelo do Whisper (`small`) baixa sozinho para `modelos\` na primeira vez que o Max abre.

## Como rodar

- `Iniciar Max.vbs`: abre sem janela (uso normal).
- `Testar Max (com janela).bat`: abre mostrando erros.
- O log fica em `max.log`.

## Configuração

Tudo fica em `config.json` (copiado do `config.exemplo.json`). Os principais:

- `voz`, `velocidade_da_voz`, `tom_da_voz`: voz do Edge.
- `modelo_whisper`, `segundos_de_silencio_para_encerrar_a_frase`, `sensibilidade_do_microfone_1a3`: escuta.
- `nota_de_perfil`: caminho de uma nota em Markdown com o seu perfil, que entra no prompt. Vazio desliga.
- `confirmar_acoes_por_voz`: comandos que não só abrem coisas pedem "sim" por voz.

### Vozes opcionais

- **ElevenLabs**: crie a variável de ambiente do Windows `ELEVENLABS_API_KEY` com a sua chave e ligue `"usar_elevenlabs": true`. Nunca escreva a chave em arquivo do projeto.
- **Kokoro (local)**: baixe `kokoro-v1.0.onnx` e `voices-v1.0.bin` da release `model-files-v1.0` do repositório thewh1teagle/kokoro-onnx no GitHub, coloque em `modelos\kokoro\` e ligue `"usar_kokoro": true`.

## O que não vai para o repositório

Veja o `.gitignore`: venv, modelos, áudios, logs, `config.json` pessoal, backups antigos e qualquer arquivo de credencial.
O login do Claude fica em `~\.claude\.credentials.json`, fora desta pasta.
