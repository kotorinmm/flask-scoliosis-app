import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from function import create_matting, foreground_generator
# from keras.preprocessing.image import load_img
# from tf.keras.models import Sequential, load_model
# from tf.keras.preprocessing import image
import numpy as np


classes = ["側弯症の可能性があります。","側弯症でないと思われます。"]
image_size = 64

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./my_model.h5') #学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            save_path = os.path.join(UPLOAD_FOLDER,"matting", filename)
            create_matting(filepath, save_path)

            image_ = Image.open(filepath)
            matte = Image.open(save_path)
            foreground = foreground_generator(image_, matte)
            new_name = os.path.join(UPLOAD_FOLDER,"cut",  filename)
            foreground.save(new_name)
            img = image.load_img(new_name, grayscale=False, target_size=(image_size,image_size))
            img = image.img_to_array(img)
            
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "この画像の人物は " + classes[predicted] + " が、断定はできないので、形成外科など然るべき医療機関を受診して下さい。"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)