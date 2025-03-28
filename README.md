# 🤖 StarCraftII DQN Agent
### **딥러닝 기반 강화학습 에이전트를 활용한 스타크래프트 II 자동화 프로젝트**
> PyTorch 기반 DQN(Deep Q-Network) 알고리즘을 활용하여 StarCraft II에서 전략을 학습하는 강화학습 에이전트 구현

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>
<img src="https://img.shields.io/badge/StarCraft2-000000?style=flat&logo=blizzardentertainment&logoColor=white"/>
<img src="https://img.shields.io/badge/ReinforcementLearning-blue?style=flat"/>

---

## 🗂 목차  
- [1. 프로젝트 개요](#1-프로젝트-개요)  
- [2. 강화학습 설계](#2-강화학습-설계)  
- [3. 주요 기능 및 코드](#3-주요-기능-및-코드)  
- [4. 학습 결과 시각화](#4-학습-결과-시각화)  
- [5. 실행 환경 및 구성](#5-실행-환경-및-구성)  
- [6. 프로젝트 기여도](#6-프로젝트-기여도)  
- [7. 연락처](#7-연락처)

---

## 1. 🎮 프로젝트 개요  

**StarCraftII-Agent**는 딥마인드의 PySC2 환경에서 작동하는 **강화학습 기반 전략 에이전트**입니다.  
복잡한 전략 게임인 스타크래프트 II 내에서 DQN(Deep Q-Network)을 기반으로 에이전트가 최적의 행동을 선택할 수 있도록 설계했습니다.

![317980205-4b6f1d97-b705-4b8d-a507-b13d6550cf10](https://github.com/user-attachments/assets/afbf3071-b302-4d00-ab57-083e5c5d685d)

---

## 2. 🤖 강화학습 설계  

### ✅ 왜 DQN을 선택했는가?

StarCraft II는 **상태 공간이 매우 크고 복잡한 환경**입니다.  
전통적인 Q-Learning은 상태-행동 쌍을 테이블 형태로 저장하므로, 상태 수가 많은 환경에서는 메모리 및 일반화 측면에서 비효율적입니다.

> 따라서 Q값을 근사화할 수 있는 **신경망 기반 Q 함수**를 사용하는 DQN이 적합하다고 판단하였습니다.

- 딥러닝 모델로 **Q(s, a)** 를 근사 → 상태 공간 확장에 유연  
- 전략 게임 내에서 다양한 시나리오에 대해 일반화된 정책 학습 가능  

---

### ✅ 왜 ε-Greedy 탐험 정책을 사용했는가?

학습 초기에는 최적의 행동에 대한 정보가 부족하므로, **무작위 행동(탐험)** 을 해야 합니다.  
하지만 후반에는 이미 학습된 정보를 바탕으로 **최적의 행동(활용)** 을 수행하는 것이 중요합니다.

> 이 균형을 위해 **ε-Greedy 탐험 정책**을 사용하였습니다.

- ε 확률로 무작위 행동 선택 → 충분한 탐험  
- (1 - ε) 확률로 Q값이 높은 행동 선택 → 점진적 최적화  
- ε 값은 학습이 진행됨에 따라 **지속적으로 감소**시켜 탐험 비중을 줄임

```python
if self.episode_count >= 150:
    self.dqn.epsilon *= 0.999  # 탐험 감소
```
---

### ✅ 왜 EMA(Exponential Moving Average)를 사용했는가?

강화학습의 보상은 에피소드마다 편차가 클 수 있기 때문에, 최근 성능의 변화 추세를 부드럽게 관찰할 필요가 있습니다.

> 따라서 **EMA(지수 이동 평균)** 을 사용하여 에이전트 성능의 흐름을 안정적으로 추적했습니다.

- 급격한 보상의 변동에도 민감하지 않고, 전반적인 성능 추세 확인 가능

- 학습이 안정적으로 진행되고 있는지 모니터링 용도로 사용

```python
self.ema = EMAMeter(factor=0.5)
self.ema.update(self.cum_reward)
```
---

## 3. 🛠 주요 기능 및 코드

### 🎮 행동 정의 (Action Space)

에이전트가 선택할 수 있는 행동은 총 6가지이며, 각각의 행동은 PySC2 RAW API를 기반으로 실제 유닛 조작을 수행합니다.

| 행동 명령 | 설명 |
|----------|------|
| `do_nothing` | 아무것도 하지 않음 |
| `harvest_minerals` | SCV를 선택하여 가장 가까운 미네랄 채취 |
| `build_supply_depot` | 자원이 충분할 때 SCV로 서플라이 디포 건설 |
| `build_barracks` | 서플라이가 완료되면 SCV로 배럭 건설 |
| `train_marine` | 배럭에서 마린 생산 |
| `attack` | 마린을 선택해 적 기지 방향으로 공격 수행 |

```python
self.actions = (
    "do_nothing",
    "harvest_minerals",
    "build_supply_depot",
    "build_barracks",
    "train_marine",
    "attack"
)
```
---

### 🧠 상태 벡터 구성 (State Representation)

21차원의 상태 벡터는 현재 게임 상황을 수치로 요약한 정보이며, DQN 신경망의 입력값으로 사용됩니다.

```python
state = (
    len(scvs),
    len(idle_scvs),
    len(command_centers),
    len(marines),
    queued_marines,
    free_supply,
    can_afford_supply_depot,
    ...
    len(enemy_barrackses),
    len(enemy_marines)
)
```
- 자원 상태, 유닛 수, 생산 가능 여부, 적의 유닛 정보 등을 포함

- PySC2의 raw_units를 직접 활용하여 세부 유닛 상태 접근
---
### 🔁 학습 루프 (Step 함수)

게임 환경에서 매 스텝마다 수행되는 에이전트의 행동 결정 및 학습 로직입니다.

```python
def step(self, obs):
    state = self.get_state(obs)
    action_idx = self.dqn.choose_action(state)
    action = self.actions[action_idx]
    
    if previous_action is not None:
        self.dqn.learn(previous_state, previous_action, reward, current_state, done)
```
- 상태(state)를 기반으로 DQN으로부터 행동(action)을 선택

- 이전 상태/행동/보상 정보를 바탕으로 Q-network 업데이트 수행

- 학습 종료 시 모델 저장 및 보상 통계 기록
---
### 📚 DQN 학습 구조
```python
self.qnetwork = NaiveMLP(input_dim=21, output_dim=6, ...)
self.dqn = NaiveDQN(
    state_dim=21,
    action_dim=6,
    qnet=self.qnetwork,
    epsilon=0.9,
    gamma=1.0,
    lr=1e-4
)
```
- 입력: 21차원 상태

- 출력: 6개의 행동에 대한 Q값

- 옵티마이저: Adam

- 활성화 함수: ReLU

- 출력층: Identity (Q값 그대로 출력)
---
### 📉 탐험 감소 및 EMA 적용
```python
if self.episode_count >= 150:
    self.dqn.epsilon *= 0.999  # 탐험 비율 점진적 감소

self.ema.update(self.cum_reward)  # 보상 추세 추적
```
- ε-Greedy 정책을 통해 탐험-활용 균형 제어

- EMA (지수 이동 평균) 으로 누적 보상의 안정적 평균 추적

---
## 4. 📊 학습 결과 시각화

학습이 진행됨에 따라 성능을 시각화하여 에이전트의 정책이 실제로 얼마나 효과적인지 확인했습니다.

### 📈 누적 점수 변화 (Cumulative Score per Episode)

에피소드마다 기록된 누적 점수 (게임 내 score_cumulative)를 시각화합니다.  
→ 보상이 점점 증가하는 모습을 통해 에이전트의 전략이 향상됨을 확인할 수 있습니다.

```python
def plot_results(self):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(self.cumulative_scores) + 1),
             self.cumulative_scores,
             label='Cumulative Score',
             linestyle='-',
             marker='o')
    plt.title('Cumulative Score over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.show()
```
![image (15)](https://github.com/user-attachments/assets/176f6031-b6fa-486f-a5e7-a2c0181dbceb)

---

### 🟩 승/무/패 비율 변화 (Win / Draw / Defeat Rate)

학습이 진행되면서 각각의 에피소드에 대한 승률, 무승부율, 패배율을 기록하고, 누적 통계로 시각화합니다.

→ 에이전트의 전반적인 전략 효율성과 안정성을 평가하는 데 사용됩니다.

---

### 📌 시각화 요약
- ✅ 승률이 80% 이상 도달하면서 정책이 효과적으로 학습됨을 확인

- ✅ 무/패 비율 감소 → 공격 타이밍, 유닛 생산, 자원 운영 전략이 점점 최적화

- ✅ 그래프 기반 실험 검증을 통해 신뢰성 있는 모델 평가 수행
