@echo off
REM AIPA 데이터 파이프라인 자동 실행 (매일 스케줄러용)
REM API 한도 소진될 때까지 무한 반복, 에러 발생 시 탈출
cd /d C:\Users\user\Git\AIPA_Engine
set COUNT=0
:LOOP
set /a COUNT+=1
echo [%COUNT%] %date% %time% >> data\pipeline\pipeline.log
python data\scripts\pipeline.py --source naver >> data\pipeline\pipeline.log 2>&1
if %ERRORLEVEL% NEQ 0 goto END
goto LOOP
:END
echo [DONE] API limit reached after %COUNT% runs. %date% %time% >> data\pipeline\pipeline.log
