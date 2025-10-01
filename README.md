# 🧪 GBP 기반 Simulation-Based Optimization 예제

## 📌 프로젝트 개요
본 예제는 **DEVS 기반 시뮬레이션 모델**과 **강화학습(Proximal Policy Optimization, PPO)** 알고리즘을 결합하여  
다중 서버 생산 시스템의 **운영 파라미터를 최적화**하는 프로젝트입니다.

DEVS(Discrete Event System Specification)는 생산 및 서비스 시스템의 동적 거동을 정밀하게 모델링할 수 있는 시뮬레이션 기법이며,  
본 프로젝트에서는 `PythonPDEVS` 라이브러리를 활용하여 생산 라인의 핵심 구성요소(Generator, Buffer, Processor, Collector)를 모델링했습니다.

---

## 🏗 시스템 구성

GBP 시스템은 다음 4가지 원자(Atomic) 모델로 구성된 복합(Coupled) DEVS 모델입니다.

| 컴포넌트 | 역할 |
|----------|------|
| Generator | 일정 간격(interarrival)으로 작업을 생성하며, 각 작업에는 처리에 필요한 크기(size)와 도착 시간이 포함됨 |
| Buffer | 다중 Processor 앞단에 위치하여 작업을 대기시키고, 유휴 Processor가 생기면 즉시 작업을 전달 |
| Processor | 작업을 받아 service rate에 따라 처리하고, 완료 후 Collector로 전달 |
| Collector | 완료된 작업을 수집해 평균 체류시간, 대기시간, 처리율 등 성능 지표를 계산 |

Processor 개수는 파라미터로 설정할 수 있으며, CoupledDEVS 구조를 통해 여러 Processor를 병렬로 구성할 수 있습니다.

---

## ⚙️ 환경 설정

1. `PythonPDEVS` 저장소를 클론 후 Python 3.X 환경에서 동작하도록 `Solver.py` 및 `Controller.py` 수정  
2. `src` 디렉토리에서 `pip install -e .` 명령으로 로컬 개발 환경에 설치  
3. `main.py` 실행 또는 각 컴포넌트 파일을 통해 GBP 시스템 시뮬레이션 및 PPO 학습 수행

---

## 📂 프로젝트 파일 설명

| 파일명 | 설명 |
|--------|------|
| **`xdevs_Gen.py`** | **Generator 모델** 정의. 일정한 간격(interarrival time)으로 작업(Job)을 생성하며, 각 작업에는 처리에 필요한 크기(size)와 도착 시간이 할당됨. 생산 라인의 작업 유입을 담당하는 핵심 모듈. |
| **`xdevs_Buffer.py`** | **Buffer(대기열) 모델** 정의. 여러 Processor 앞단에 위치하여 도착한 작업을 대기시키고, 유휴 Processor가 생기면 즉시 작업을 전달. 다중 서버 환경에서 처리 효율을 높이는 핵심 컴포넌트. |
| **`xdevs_Proc.py`** | **Processor 모델** 정의. Buffer로부터 Job을 받아 설정된 service rate에 따라 처리하고, 완료 시 Collector로 작업을 전달. 병렬 Processor 구성이 가능하며 시스템 처리율에 직접적인 영향을 미침. |
| **`xdevs_Coll.py`** | **Collector 모델** 정의. Processor에서 완료된 Job을 수집하고, 평균 체류시간, 대기시간, 처리율 등 **핵심 성능 지표(metrics)**를 계산하여 강화학습 보상 신호로 활용. |
| **`xdevs_Jobm.py`** | **Job 메시지 구조 및 데이터 처리 로직** 정의. Generator에서 생성되는 Job 객체의 속성(도착 시각, 크기, 상태 등)을 관리하며, 각 컴포넌트 간 Job 전달 시 공통 인터페이스 역할. |
| **`xdevs_Coupled.py`** | 위의 Generator, Buffer, Processor, Collector를 하나의 **Coupled DEVS 시스템**으로 연결하는 구성 파일. Processor 개수와 Buffer 크기 등 파라미터에 따라 동적으로 시스템 구조를 설정하며, 시뮬레이션 실행의 엔트리 포인트 역할. |
| **`PPO.py`** | **Proximal Policy Optimization** 강화학습 알고리즘 구현. 연속형(interarrival, service rate 등)과 이산형(buffer capacity, server 수) 파라미터를 동시에 제어할 수 있는 정책을 정의하고, Advantage 계산 및 clipping으로 학습 안정성 확보. |
| **`main.py`** | 전체 실행 파이프라인을 구성하는 메인 스크립트. PPO 에이전트를 초기화하고, GBP 시뮬레이션 환경을 불러와 학습 및 최적화를 진행하며, 최적 파라미터와 결과를 출력. 프로젝트의 실행 시작점(entry point). |

---

## 🧠 강화학습 적용 (PPO)

GBP 모델의 운영 파라미터는 **연속형(interarrival, service rate, job size)** 과  
**이산형(buffer capacity, server 수)** 변수를 모두 포함하므로,  
이를 동시에 제어할 수 있는 정책 기반 강화학습 알고리즘인 **PPO (Proximal Policy Optimization)** 를 사용했습니다.

### 알고리즘 선택 이유
- 연속형·이산형 파라미터를 동시에 제어 가능  
- 정책 업데이트 폭을 제한해 학습 안정성 확보  
- 시뮬레이션 실행 비용이 큰 환경에서도 샘플 효율적 활용 가능

### 학습 과정
1. PPO 에이전트가 연속/이산 행동을 샘플링  
2. 샘플링된 파라미터로 GBP 시뮬레이션 실행  
3. Collector에서 계산된 성능지표(평균 체류시간, 처리율, 총 소요시간 등)를 기반으로 보상 계산  
4. Advantage와 clipping 기법을 이용하여 정책 파라미터(mu, sigma, logits) 업데이트

---

## 📊 최적화 전·후 성능 비교

강화학습(PPO)을 통한 파라미터 최적화 전·후 시스템 성능을 비교한 결과는 다음과 같습니다.

<p align="center">
  <img src="./16c60190-0805-4e3d-b72a-a55204f454ed.png" alt="PPO 최적화 전후 성능 비교" width="750"/>
</p>


결과적으로, 단순 시뮬레이션 실행에 비해 **다목적 통합 성능 지표를 약 201% 향상**시켰으며,  
강화학습이 복합적인 성능 목표(체류시간 최소화, 처리율 최대화, 이용률 안정화 등)와 자원 제약을 동시에 고려하여  
효율적인 운영 방안을 탐색할 수 있음을 확인했습니다.

---

## 📝 요약

| 항목 | 내용 |
|------|------|
| 시뮬레이션 프레임워크 | PythonPDEVS (DEVS 기반 DES 모델링) |
| 최적화 대상 | Interarrival, Service Rate, Job Size, Buffer Capacity, Server 수 |
| RL 알고리즘 | PPO (연속 + 이산 동시 제어) |
| 최적화 목표 | 체류시간 최소화, 처리율 극대화, 비용 최소화, 이용률 안정화 |
| 성능 개선 | 통합 성능지표 약 201% 향상 |

---

## 📚 참고
- PythonPDEVS: [https://github.com/capocchi/PythonPDEVS](https://github.com/capocchi/PythonPDEVS)  
- PPO 알고리즘: Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347  
