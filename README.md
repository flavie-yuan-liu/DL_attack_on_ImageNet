# DL_attack_on_ImageNet


Implementation of attack learning algorithm based on dictionary leanring, specially for the Imagenet Dataset. 
<p align="center">
  <img src="https://github.com/flavie-yuan-liu/DL_attack_on_ImageNet/blob/main/attack_samples.png" width="650", title="ADiL">
</p>

> We have the trained model on 6 deep model ([download here](https://drive.google.com/drive/folders/1P-raQEaFttcv81q2JoujWTXwez440cQM?usp=sharing)).
> To see a simple example of adversarial image generation with our model (as shown in above), just run
```
python main.py
```

> You can also train your own model. 
```
python demo_dL_attack.py --model densenet

```
There are some parameters to regulate in ADiL attack, make your settings to have the optimal ones. 


## Reference:
__________________________________
Adversarial Dictionary Learning




