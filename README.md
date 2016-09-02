# ContextWordVectors
This is a context based word vector tool
#LDA
This is a c++ version of LDA implementation
`It is not the traditional LDA with P(w|z) and P(z|d).`
When applying this kind of LDA implementation, every word will be assigned with a fixed topic with top possibility among all the possible topics.
##Usage
###Linux Users
Open `LDA` folder and go to `src` folder, then
`make`.
When it is finished, prepare your training data.
You can find detailed usage in `docs` folder
###Windows Users
Open `LDA-Windows` folder with visual studio 2013/2015
And please refer to the `docs` folder for more detailed usage. All the command in Linux can be transformed to windows.
`Please check in the `main` function to set all the parameters right. `

There will be a new wordmap file for further word similarity task

#Word Vector
This is a modified wordvector tool by Gensim.
In order to use this, All the packages listed below should be installed with correct versions
* gensim(0.12.4)
* numpy(1.1.10)
* six(1.10.0)
* cython(0.23.5)
* `scipy(0.15.1)`

##Usage
###Linux Users
Please edit the `train.py` in the root folder with `wordmapfile` and `tassignfile` generated by LDA.
And then just run it.
###Windows Users
####Important notes:
`Please Install visual studio 2008 Professional to meet the required compiler if you are using python2.7`
Visual studio is highly recommended to edit and debug python on windows.
Create a python solution and add the whole `WordVector` folder to your solution directory. And please edit the `train.py` in the root folder with `wordmapfile` and `tassignfile` generated by LDA.

#WordSimilarity
The dataset is `SCWS`. And all the data in the `WordSimilarity` is just a demo, you need to use `ratings.txt` in the folder as your original dataset.
##Python package Requiements
* numpy
* NLTK
* enchant
* scipy

##Usage
you need to go through the `solveLDA.py` in the folder and pay attention to the directories shown in the code in order to make it run correctly.

#Text Classification
The dataset is `20NewsGroup`. You can download this dataset in `http://qwone.com/~jason/20Newsgroups/`. I recommend the 20news-bydate.tar.gz because it has divided the data into training part and testing part.
The code is written in Python.
##Usage
You need install scikit-learn package.
First of all,you need run LDA method to generate inference files. Put them together with training and testing data.
There are 3 file .py files. You need to extract the .tar.gz file in the current "TextClassification/" folder. And then run ForLDAUse.py->Preprocess.py->svm.py one after another.

If met with any problem, please contact me:`wxkdesky@hotmail.com`
Good luck!