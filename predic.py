from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
def predic(images: Image.Image):
    model=load_model('before_finetune.h5')
    print("type in pred: ",type(images))
    testc=images.resize((220,220))
    test_img=image.img_to_array(testc)
    np_image = np.array(test_img).astype('float32')/255
    test_final_image=np.expand_dims(np_image,axis=0)
    #image = np.asarray(image.resize((224, 224)))[..., :3]
    result=model.predict(test_final_image)
    ind=np.argmax(result)
    name_dic={'adonis': 0,
    	 'american snoot': 1,
	 'an 88': 2,
	 'banded peacock': 3,
	 'beckers white': 4,
	 'black hairstreak': 5,
	 'cabbage white': 6,
	 'chestnut': 7,
	 'clodius parnassian': 8,
	 'clouded sulphur': 9,
	 'copper tail': 10,
	 'crecent': 11,
	 'crimson patch': 12,
	 'eastern coma': 13,
	 'gold banded': 14,
	 'great eggfly': 15,
	 'grey hairstreak': 16,
	 'indra swallow': 17,
	 'julia': 18,
	 'large marble': 19,
	 'malachite': 20,
	 'mangrove skipper': 21,
	 'metalmark': 22,
	 'monarch': 23,
	 'morning cloak': 24,
	 'orange oakleaf': 25,
	 'orange tip': 26,
	 'orchard swallow': 27,
	 'painted lady': 28,
	 'paper kite': 29,
	 'peacock': 30,
	 'pine white': 31,
	 'pipevine swallow': 32,
	 'purple hairstreak': 33,
	 'question mark': 34,
	 'red admiral': 35,
	 'red spotted purple': 36,
	 'scarce swallow': 37,
	 'silver spot skipper': 38,
	 'sixspot burnet': 39,
	 'skipper': 40,
	 'sootywing': 41,
	 'southern dogface': 42,
	 'straited queen': 43,
	 'two barred flasher': 44,
	 'ulyses': 45,
	 'viceroy': 46,
	 'wood satyr': 47,
	 'yellow swallow tail': 48,
	 'zebra long wing': 49}
    name=list(name_dic)
    return name[ind], str(result[0][ind])
