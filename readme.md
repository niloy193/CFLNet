
Setup: Run

  ```shell
  pip install -r requirements.txt
  ```

Step 1: Download IMD-20 Real Life Manipulated Images from [Link](http://staff.utia.cas.cz/novozada/db/).

step2: set the dataset path in  
 ``` config.py - base_dir ``` 
for example, if you have downloaded and unzipped the IMD2020 dataset in the following directory:'/home/forgery/' then put '/home/forgery/' as the base_dir. (DO NOT put '/home/forgery/IMD2020/)

Step 3: To train the model run   
  ```shell
  python trainer.py
  ```

step 4: get the pretrained model from [here](https://drive.google.com/drive/folders/1pjPBNMqTwK33KLEkLv3UZNXM2wWX2-PR?usp=sharing)


Step 5: To evaluate the model after training, run

  ```shell
   python evaluate.py --pretrained_model  imd_2020_best_model.pth
  ```
