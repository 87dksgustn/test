# Surrogate Predictor 현업용 1페이지 실행 가이드

## 1) 목적
- 이 도구는 배터리 설계 입력값으로 TP/NoTP 및 예측 결과를 빠르게 확인하는 로컬 실행 도구입니다.
- 서버 구축 없이 PC에서 바로 실행합니다.

## 2) 준비물
- 전달받은 파일: SurrogatePredictor_portable_vXXXX.X.zip
- 운영체제: Windows 10/11
- 인터넷: 없어도 실행 가능

## 3) 실행 방법 (3단계)
1. ZIP 파일을 사용자 쓰기 가능한 폴더에 압축 해제
- 권장: 바탕화면, 문서, D 드라이브 작업 폴더
- 비권장: Program Files 같은 관리자 권한 폴더

2. SurrogatePredictor.exe 실행
- 첫 실행 시 보안 경고가 뜨면 사내 정책에 따라 허용
- 정상 실행되면 브라우저가 자동으로 열림

3. 화면이 열리지 않으면
- 브라우저 주소창에 http://127.0.0.1:8501 입력

## 4) 사용 방법
### 단건 예측
1. Single Prediction 영역에서 입력값 입력
2. Predict 클릭
3. 결과 표에서 predicted_label, 확률, 예측값 확인

### 일괄 예측 (CSV)
1. Batch Prediction 영역에서 CSV 업로드
2. Run Batch Prediction 클릭
3. 결과 확인 후 Download Predictions CSV 클릭

필수 컬럼명:
- A_Cell_D
- C_Barrier_Thx
- E_Barrier_Outer_Thx
- F_ThermalResin_Thx
- B_Barrier_Type
- D_Barrier_Outer_Type

## 5) 이력 저장 옵션
- 좌측 Runtime Options의 Save prediction history로 ON/OFF 선택
- 기본값: OFF
- ON일 때만 outputs/prediction_history.csv에 누적 저장

## 6) 모델 파일(번들) 정책
- 기본 내장 모델 자동 사용
- 필요 시 화면의 Bundle path에서 다른 번들 파일로 변경 가능

## 7) 자주 발생하는 문제
1. EXE를 눌러도 반응이 없음
- 같은 폴더의 launcher.log 확인
- 보안 프로그램 차단 여부 확인

2. Bundle not found 오류
- 기본 파일 누락 여부 확인
- Bundle path를 올바른 pkl 파일로 지정

3. CSV 업로드 후 실패
- 필수 컬럼명 오탈자 확인
- 숫자 컬럼에 문자/빈값 없는지 확인

## 8) 현업 전달 체크리스트
- ZIP 파일 버전명 확인
- 샘플 입력 CSV 1개 동봉
- 실행 스크린샷 1장 동봉
- 문의 담당자/연락처 동봉

## 9) 배포 담당자 메모
- 코드서명 배포본이면 보안 경고가 줄어듭니다.
- 분기 배포 시 파일명에 버전을 고정하세요. 예: SurrogatePredictor_portable_v2026.3.zip
