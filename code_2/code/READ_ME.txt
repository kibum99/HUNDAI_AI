아래와 같은 환경에서 코드를 실행해야 합니다.
* Anaconda3가 다운로드 되어있어야 합니다.
* pytorch는 1.10.2ver 입니다.
  (다운로드 : conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch)
* 압축이 해제된 code폴더의 위치는 다음과 같습니다.
F:
├── Competition_No1
└── Competition_No2
    ├── code (*)
    └── datasets
         ├── test
         │   ├── 01
         │   │   ├── gt_test_01.txt
         │   │   └── images
         │   ├── 02
         │   └── 03
	 └── train

조건이 모두 갖춰진 후에 아래의 방법으로 결과를 확인할 수 있습니다.
1. Anaconda Prompt 를 실행합니다.
2. 아래 3줄을 그대로 복사하여 Anaconda prompt에 실행합니다.

F:
cd F:\Competition_No2\code
python Run_Test.py

3. code 폴더 내 GTX.txt와 result.csv를 실행하여 결과를 확인합니다.