# PowerShell 実行スクリプト
# main.py を Python で起動し、GUI ウィンドウを表示させる

# Python 実行ファイルのパス（環境に合わせて修正してください）
$PythonPath = "C:\Users\Kaito_Tanaka\AppData\Local\Programs\Python\Python313\python.exe"

# main.py のパス
$ScriptPath = "C:\Users\Kaito_Tanaka\Desktop\private_file\kancolle_data\kancolle_program\main.py"

# trees.py のパス(階層構造を記述するpyファイル)
$Trees_Path = "C:\Users\Kaito_Tanaka\Desktop\private_file\kancolle_data\kancolle_program\tools\treegen\treegen.py"

# concern_audit.pyのパス(関心数の記録用ファイル)
$Concern_Audit_Path = "C:\Users\Kaito_Tanaka\Desktop\private_file\kancolle_data\kancolle_program\tools\concern_audit\concern_audit.py"

# 実行
& $PythonPath $Concern_audit_path
& $PythonPath $Trees_path
& $PythonPath $ScriptPath
