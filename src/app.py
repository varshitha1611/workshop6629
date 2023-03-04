from flask import Flask, request,jsonify
from mynn import MyFirstNN
app=Flask(__name__)

NN = MyFirstNN()

@app.route('/api/myfirstnn',methods=['GET','POST'])

def mynn():
    if request.method=='GET':
        return jsonify("this is my nn")
    else:
        input_json=request.json
        nn_inputs=input_json['input']
        output=NN.predict(nn_inputs)
        return jsonify(output)
    
if __name__=='__main__':
    app.run('0.0.0.0',port=8401)
