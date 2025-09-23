[Setup]
AppName=Vision-Vibe
AppVersion=1.0
DefaultDirName={pf}\Vision-Vibe
DefaultGroupName=Vision-Vibe
OutputDir=output
OutputBaseFilename=VisionVibeInstaller
Compression=lzma
SolidCompression=yes

[Files]
; Copy toàn bộ dist/main vào thư mục cài đặt
Source: "dist\main\*"; DestDir: "{app}"; Flags: recursesubdirs
Source: "app.ico"; DestDir: "{app}"

[Icons]
; Shortcut trong Start Menu
Name: "{group}\Vision-Vibe"; Filename: "{app}\main.exe"; IconFilename: "{app}\app.ico"
; Shortcut ngoài Desktop
Name: "{commondesktop}\Vision-Vibe"; Filename: "{app}\main.exe"; IconFilename: "{app}\app.ico"

