# face-recognition

This project combines the best ideas from [Skuldur/facenet-face-recognition](https://github.com/Skuldur/facenet-face-recognition) and [Sefik Ilkin Serengil](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/). Thank you for writing awesome code and tutorials!

This repository contains a demonstration of face recognition using the [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) network  and one-shot learning. It uses a pretrained model to encode the images and uses Euclidean distance to find the difference. If you need a quick introduction to FaceNet and one-shot learning, refer this: [Making your own Face Recognition System](https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c).

## Up and Running

To install all the requirements for the project run

	pip install -r requirements.txt

Download the model, [facenet_keras.h5](https://www.dropbox.com/s/xwn57bffg5xobb8/facenet_keras.h5?dl=1) and put it in `models` directory.
	
Keep the face images of people you want to recognize in `databases` directory. 

Keep the input photo (on which you want to run the program) in `input` directory.

In the root directory. After the modules have been installed you can run the project by using python

	python facenet.py
	
And see the outputs in `output` directory
