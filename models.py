from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Embedding, 
    Activation, 
    Flatten, 
    Dense, 
    concatenate,
    Conv1D,
    MaxPooling1D,
    Dropout,
    GlobalMaxPooling1D,
    BatchNormalization,
    Average
    )
from keras import optimizers

def singleModel():
    model = Sequential()
    model.add(Input(shape=(3,107)))
    model.add(Conv1D(150,3,strides=1,padding='same', activation="relu"))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(BatchNormalization()) 
    model.add(Dropout(0.5))
    model.add(Dense(150,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1,))
    model.compile(loss="mse", optimizer=optimizers.SGD(learning_rate=0.01,momentum=0.6,nesterov=True))
    return model

def multiModel(singleModel_): 
    target1 = Input(shape=(1,107),name="target1")
    target2 = Input(shape=(1,107),name="target2")
    leftC = Input(shape=(1,107),name="leftC")
    rightC = Input(shape=(1,107),name="rightC")
    inputs = [leftC,target1,target2,rightC] 

    lcAvg = Average()([leftC,target1])
    rcAvg = Average()([target2,rightC])
    targetInputL = concatenate([leftC,target1,rcAvg],axis=1)
    targetInputR = concatenate([lcAvg,target2,rightC],axis=1)

    CovL = Conv1D(214,3,padding="same",activation="relu")
    CovR = Conv1D(214,3,padding="same",activation="relu")
    Lemb = MaxPooling1D(pool_size=3)(CovL(targetInputL))
    Remb = MaxPooling1D(pool_size=3)(CovR(targetInputR))

    Lemb = Dense(107,activation="relu")(Dropout(0.3)(Lemb))
    Remb = Dense(107,activation="relu")(Dropout(0.3)(Remb))

    targetInput = concatenate([Lemb,Remb],axis=-1)

    squashTarget = Dense(214,activation="relu")(Dropout(0.5)(targetInput))
    squashTarget = Dense(107,activation="relu")(Dropout(0.5)(squashTarget))

    mainInput = concatenate([leftC,squashTarget,rightC],axis=1, name="MainInput")
    preds = singleModel_(mainInput)
    model = Model(inputs=inputs, outputs=preds)
    model.compile(loss="mse", optimizer=optimizers.Adam())
    return model  