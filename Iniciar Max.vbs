' Abre o Max sem janela preta. De dois cliques para iniciar.
Set sh = CreateObject("WScript.Shell")
pasta = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = pasta
sh.Run """" & pasta & "\.venv\Scripts\pythonw.exe"" """ & pasta & "\max.py""", 0, False
