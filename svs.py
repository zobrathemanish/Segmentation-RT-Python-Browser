__author__ = 'Nirav'
#
from flask import Flask,render_template, url_for,request,redirect,flash,Response
from datetime import datetime
#from logging import DEBUG
from camera import VideoCamera
import os

#Calls flask constructor for global application object
app = Flask(__name__)
#app.logger.setLevel(DEBUG)
app.config['SECRET_KEY']='\x98.\x80\xba1\xcc\x0cU\xd0\xdb\xd8\x9c8\x0e\xb1EgA\xb6\xde\x84\xcby\xf8\xb5\xed\xe5E\xaav\xed\x16'
userregister = []

def storuser(em,pass1,pass2,fname,lname):
    userregister.append(dict(email=em,password1=pass1,password2=pass2,FirstName=fname,LastName=lname,datet = datetime.utcnow()
                             )
                        )

#This is view function
@app.route('/')
@app.route('/index')
@app.route('/Index')
#This your view
def index():
    #render template render HTML template.
     return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/stream')
def stream():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('stream.html')


@app.route('/reguser',methods = ['GET','POST'])
def reguser():
    if request.method == "POST":
        femail = request.form['txtemail']
        fpass1 = request.form['txtpass1']
        fpass2 = request.form['txtpass2']
        ffname = request.form['txtfname']
        flname = request.form['txtlname']
        storuser(femail,fpass1,fpass2,ffname,flname)
        flash("User Registration Done:{}".format(femail))
        #app.logger.debug("New register user detail:-"+fpass1)
        return redirect(url_for('sucess'))
    return render_template('UserRegister.html')

@app.errorhandler(500)
def server_err(e):
    return  render_template("500.html"),500

@app.errorhandler(404)
def pagenot_found(e):
    return  render_template("404.html"),400

if __name__ == '__main__':
    app.run(host=os.getenv('IP',0.0.0.0'),port=int(os.getenv('PORT',5000)))