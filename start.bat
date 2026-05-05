@echo off
echo Starting Chicken Wing Counting System...
start /B docker compose up --build

echo Waiting for server to be ready...
:loop
timeout /t 2 /nobreak >nul
curl -s -o nul -w "%%{http_code}" http://localhost:5580 | findstr "200" >nul
if errorlevel 1 goto loop

echo Server is ready! Opening browser...
start http://localhost:5580
