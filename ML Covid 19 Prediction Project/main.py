from flask import Flask, render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()



@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = (request.form)
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])

        #code for infarence
        inputFeatures = [fever, bodyPain, age, runnyNose, diffBreath]
        infprob = clf.predict_proba([inputFeatures])[0][1]
        #return 'Hello, World!' + str(infprob)

        print(infprob)
        return render_template('show.html', inf=round(infprob*100))
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)