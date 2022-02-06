# Module 구현 목록
- 각 모듈은 Keras의 call, pyTorch의 forward와 같은 forward 함수가 존재함
- backpropagation을 위한 backward가 존재함. 
- Gradient는 멤버변수로 self.tape에 저장 되며 각 레이어의 특성에 따라 정의함

