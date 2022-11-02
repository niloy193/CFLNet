
Setup: Run

  ```shell
  pip install -r requirements.txt
  ```

Step 1: Download IMD-20 Real Life Manipulated Images from [Link](http://staff.utia.cas.cz/novozada/db/).

step2: set the dataset path in  

for example, if you have downloaded and unzipped the IMD2020 dataset in the following directory: ``` /home/forgery/ ```  then put  ``` /home/forgery/ ```  as the base_dir  in the config file. (DO NOT put  ``` /home/forgery/IMD2020/ ``` in base_dir )

Step 3: To train the model run   
  ```shell
  python trainer.py
  ```
