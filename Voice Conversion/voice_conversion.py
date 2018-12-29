from flask import Flask, render_template, request, redirect
import os
app = Flask(__name__)


@app.route('/')
def mainpage():
    return render_template('mainpage.html')


@app.route('/mainpage')
def mainpage_():
    return render_template('mainpage.html')


@app.route('/record', methods=['GET', 'POST'])
def record():
    return render_template('record.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')


@app.route('/uploadsth', methods=['GET'])
def uploadsth():
    filename = str(request.values.get('data'))
    print('filename: ' + filename)
    picpath1 = "images/blank.jpg"
    # picpath2 = picpath1
    if(filename != 'None'):
        portion = os.path.splitext(filename)
        if portion[1] == ".wav" or ".WAV":
            picname = portion[0] + ".jpg"
            # filepath = 'http://127.0.0.1:5000/static/audio/' + filename
            filepath = './static/audio/' + filename
            print('filepath: ' + filepath)
            picpath1 = 'images/' + picname
            if not os.path.exists('./static/' + picpath1):
                cmd = 'python draw_wave.py ' + filepath
                print('cmd: ' + cmd)
                os.system(cmd)
            else:
                print('exist')
    # return render_template('upload.html', picpath1 = picpath1, picpath2 = picpath2)
    return render_template('upload.html')

@app.route('/recordsth', methods=['GET'])
def recordsth():
    filename = str(request.values.get('data'))
    print('filename: ' + filename)
    picpath1 = "images/blank.jpg"
    # picpath2 = picpath1
    if(filename != 'None'):
        portion = os.path.splitext(filename)
        if portion[1] == ".wav" or ".WAV":
            picname = portion[0] + ".jpg"
            # filepath = 'http://127.0.0.1:5000/static/audio/' + filename
            filepath = './static/audio/' + filename
            print('filepath: ' + filepath)
            picpath1 = 'images/' + picname
            if not os.path.exists('./static/' + picpath1):
                cmd = 'python draw_wave.py ' + filepath
                print('cmd: ' + cmd)
                os.system(cmd)
            else:
                print('exist')
    # return render_template('upload.html', picpath1 = picpath1, picpath2 = picpath2)
    return render_template('record.html')

@app.route('/upload/convert', methods=['GET'])
def uploadconvert():
    get_data = str(request.values.get('data')).split(':')
    target = get_data[0]
    filename = get_data[1]
    print('filename: ' + filename)
    picpath1 = "images/blank.jpg"
    # picpath2 = picpath1
    if(filename != 'None'):
        portion = os.path.splitext(filename)
        if portion[1] == ".wav" or ".WAV":
            if not os.path.exists('./static/audio/convert/' + target + '/' + filename):
                # cmd = 'python ./project/final_conversion.py -case ' + target + ' -net1 model-17900 -net2 model-17900 -file ./static/audio/' + filename + ' -savepath ./static/audio/convert/' + target + '/'
                if target == 'slt':
                    cmd = 'python ./project/final_conversion.py -case slt -net1 model-17900 -net2 model-17900 -file ./static/audio/' + filename + ' -savepath ./static/audio/convert/slt/'
                elif target == 'ksp':
                    cmd = 'python ./project/final_conversion.py -case ksp -net1 model-17900 -net2 model-12000 -file ./static/audio/' + filename + ' -savepath ./static/audio/convert/ksp/'
                print('cmd: ' + cmd)
                os.system(cmd)
            else:
                print('converted before.')
            picname = portion[0] + ".jpg"
            # filepath = 'http://127.0.0.1:5000/static/audio/convert/' + target + '/' + filename
            filepath = './static/audio/convert/' + target + '/' + filename
            print('filepath: ' + filepath)
            picpath1 = 'images/convert/' + target + '/' + picname
            if not os.path.exists('./static/' + picpath1):
                cmd = 'python draw_wave.py ' + filepath
                print('cmd: ' + cmd)
                os.system(cmd)
            else:
                print('exist.')
    # return render_template('upload.html', picpath1 = picpath1, picpath2 = picpath2)
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
