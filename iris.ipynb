{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "deeb67f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2022-05-25 18:18:40.648 INFO    numexpr.utils: NumExpr defaulting to 4 threads.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn import datasets\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "06e90b2d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2022-05-25 18:20:03.625 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "st.write(\"\"\"\n",
    "# Simple Iris Flower Prediction App\n",
    "This app predicts the **Iris flower** type!\n",
    "\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "367ad25d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=1, _provided_cursor=None, _parent=DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None), _block_type=None, _form_data=None)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.sidebar.header('User Input Parameters')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f020fc10",
   "metadata": {},
   "outputs": [],
   "source": [
    "def user_input_features():\n",
    "    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)\n",
    "    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)\n",
    "    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)\n",
    "    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)\n",
    "    data = {'sepal_length': sepal_length,\n",
    "            'sepal_width': sepal_width,\n",
    "            'petal_length': petal_length,\n",
    "            'petal_width': petal_width}\n",
    "    features = pd.DataFrame(data, index=[0])\n",
    "    return features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d32916c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = user_input_features()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "64ae01a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('User Input parameters')\n",
    "st.write(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "31d52938",
   "metadata": {},
   "outputs": [],
   "source": [
    "iris = datasets.load_iris()\n",
    "X = iris.data\n",
    "Y = iris.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "12ba73f5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1add82d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = RandomForestClassifier()\n",
    "clf.fit(X, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a6e1cc5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = clf.predict(df)\n",
    "prediction_proba = clf.predict_proba(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1edf36aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('Class labels and their corresponding index number')\n",
    "st.write(iris.target_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "cbe832cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('Prediction')\n",
    "st.write(iris.target_names[prediction])\n",
    "#st.write(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9c9066ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.subheader('Prediction Probability')\n",
    "st.write(prediction_proba)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a75aafb5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
