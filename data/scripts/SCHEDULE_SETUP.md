# AIPA 파이프라인 매일 자동 실행 설정

## 1. PowerShell을 관리자 권한으로 실행

## 2. 작업 스케줄러 등록
```powershell
Set-Location C:\Users\user\Git\AIPA_Engine

schtasks /create /tn "AIPA_Pipeline_Daily" /tr "C:\Users\user\Git\AIPA_Engine\data\scripts\run_pipeline.bat" /sc daily /st 06:00 /f
```

## 3. 등록 확인
```powershell
schtasks /query /tn "AIPA_Pipeline_Daily"
```

## 4. 수동 테스트 실행
```powershell
schtasks /run /tn "AIPA_Pipeline_Daily"
```

## 5. 삭제 (필요시)
```powershell
schtasks /delete /tn "AIPA_Pipeline_Daily" /f
```

## 참고
- 매일 오전 6시에 자동 실행 (네이버 API 일일 할당량 초기화 후)
- 실행 로그: `data\pipeline\pipeline.log`
- PC가 켜져 있어야 실행됨 (절전모드 X)
- 시간 변경: `/st 06:00` 부분 수정
