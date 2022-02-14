# TF-Net
TF-Net is a deepfake detection model, considering features from self-supervised template maps, frequency domain by the DCT module, sequential correlation from the GRU module.
![img](https://github.com/serendipity109/TF-Net/blob/master/diagram.PNG)
Feel free to try the colab [demo](https://colab.research.google.com/drive/16OSk-F4Mv-E_v994SiXiM0mgQjKhz_Ip?usp=sharing).

All video rights are reserved by the owner. 

[real.mp4](https://www.youtube.com/watch?v=h45KOn8UgpY&t=1s&ab_channel=TODAY%E7%9C%8B%E4%B8%96%E7%95%8C)
[fake.mp4](https://www.youtube.com/watch?v=cQ54GDm1eL0&ab_channel=BuzzFeedVideo)

## Usage
### Docker
Pull the docker image from docker hub.
```console
$ docker pull jayda960825/tf-net
```
Clone this repository and change the directories in docker-compose.yml (~/docker_oup/video, ~/docker_oup/output ~/docker_oup/aftExt) to your local directories and run.
```console
$ docker-compose up
```
### GCP VM
Contact me through jayda960825@gmail.com to open up a firewall rule to access the api.  
Send a post request to http://34.135.228.79:8087/predict
