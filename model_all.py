#SqueezeNet 11
import tensorflow as tf
from keras_tuner import HyperModel
import tensorflow.keras.backend as K
from keras_tuner import HyperParameters

from tensorflow.keras import layers
from keras_tuner import RandomSearch
from tensorflow import keras
import numpy as np

class SqueezeNet11Model(HyperModel):
    def __init__(self, classes, IMG_WIDTH, IMG_HEIGHT):
        self.classes = classes
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def get_axis(self):
        axis = -1 if K.image_data_format() == 'channels_last' else 1
        return axis

    def create_fire_module(self,x, nb_squeeze_filter, name, use_bypass=False):
        """
        Creates a fire module
        
        Arguments:
            x                 : input
            nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
            use_bypass        : if True then a bypass will be added
            name              : name of module e.g. fire123
        
        Returns:
            x                 : returns a fire module
        """
        
        nb_expand_filter = 4 * nb_squeeze_filter
        squeeze    = layers.Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
        expand_1x1 = layers.Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
        expand_3x3 = layers.Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
        
        axis = self.get_axis()
        x_ret =  layers.Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
        
        if use_bypass:
            x_ret =  layers.Add(name='%s_concatenate_bypass'%name)([x_ret, x])
            
        return x_ret


    def output(self,x, nb_classes):
        x = layers.Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
        x = layers.GlobalAveragePooling2D(name='avgpool10')(x)
        x = layers.Activation("softmax", name='softmax')(x)
        return x


    def build(self, hp):
       
        use_bypass = hp.Boolean('use_bypass')
        compression = hp.Fixed('compression',1.0)

        input_img =  layers.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))


        x = layers.Conv2D(int(64*compression), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)

        x = self.create_fire_module(x, int(16*compression), name='fire2')
        x = self.create_fire_module(x, int(16*compression), name='fire3')

        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool3')(x)

        x = self.create_fire_module(x, int(32*compression), name='fire4')
        x = self.create_fire_module(x, int(32*compression), name='fire5')

        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool5')(x)

        x = self.create_fire_module(x, int(48*compression), name='fire6')
        x = self.create_fire_module(x, int(48*compression), name='fire7')
        x = self.create_fire_module(x, int(64*compression), name='fire8')
        x = self.create_fire_module(x, int(64*compression), name='fire9')

        dropout_rate = hp.Choice('dropout_rate',values=[0.1,0.5,0.8])

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        # Creating last conv10
        x = self.output(x, self.classes)
        model = keras.Model(inputs=input_img, outputs=x)
        optimizer = hp.Choice("optimizer", ["adam", "sgd"])
        model.compile(
            optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model


class SqueezeNetModel(HyperModel):
    def __init__(self, classes, IMG_WIDTH, IMG_HEIGHT):
        self.classes = classes
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def get_axis(self):
        axis = -1 if K.image_data_format() == 'channels_last' else 1
        return axis

    def create_fire_module(self,x, nb_squeeze_filter, name, use_bypass=False):
        """
        Creates a fire module
        
        Arguments:
            x                 : input
            nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
            use_bypass        : if True then a bypass will be added
            name              : name of module e.g. fire123
        
        Returns:
            x                 : returns a fire module
        """
        
        nb_expand_filter = 4 * nb_squeeze_filter
        squeeze    = layers.Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
        expand_1x1 = layers.Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
        expand_3x3 = layers.Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
        
        axis = self.get_axis()
        x_ret =  layers.Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
        
        if use_bypass:
            x_ret =  layers.Add(name='%s_concatenate_bypass'%name)([x_ret, x])
            
        return x_ret


    def output(self,x, nb_classes):
        x = layers.Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
        x = layers.GlobalAveragePooling2D(name='avgpool10')(x)
        x = layers.Activation("softmax", name='softmax')(x)
        return x


    def build(self, hp):
       
        use_bypass = hp.Boolean('use_bypass')
        compression = hp.Fixed('compression',1.0)
        
        input_img = layers.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))

        x = layers.Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
        
        x = self.create_fire_module(x, int(16*compression), name='fire2')
        x = self.create_fire_module(x, int(16*compression), name='fire3', use_bypass=use_bypass)
        x = self.create_fire_module(x, int(32*compression), name='fire4')
        
        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
        
        x = self.create_fire_module(x, int(32*compression), name='fire5', use_bypass=use_bypass)
        x = self.create_fire_module(x, int(48*compression), name='fire6')
        x = self.create_fire_module(x, int(48*compression), name='fire7', use_bypass=use_bypass)
        x = self.create_fire_module(x, int(64*compression), name='fire8')
        
        x =  layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool8')(x)
        
        x = self.create_fire_module(x, int(64*compression), name='fire9', use_bypass=use_bypass)

        dropout_rate = hp.Choice('dropout_rate',values=[0.0,0.5,0.8])
        
        if dropout_rate:
            x =layers.Dropout(dropout_rate)(x)
            
        x =  self.output(x, self.classes)
        model = keras.Model(inputs=input_img, outputs=x)
        optimizer = hp.Choice("optimizer", ["adam", "sgd"])
        model.compile(
            optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
 

class SqueezeNetAutoModel(HyperModel):
    def __init__(self, classes):
        self.classes = classes

    def build(self, hp):
       
          
        compression_val = hp.Fixed('compression',1.0)
        
        input_img =  layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

        
        x = layers.Conv2D(int(96*compression_val), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
         

        j = 2
        filter_size = 16

        num_fire = hp.Int("fire_module", 1,2, default=2)
        use_bypass = [ hp.Boolean('use_bypass'+str(i)) for i in range(num_fire)]
        pooling = [ hp.Choice('pooling'+str(i), ["max", "avg"]) for i in range(num_fire) ]
        
        print(use_bypass)
        print(pooling)
        for i in range(num_fire):
         
             
            x = create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j), )

            x = create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j+1),use_bypass=use_bypass[i])

            #if hp.Choice("pooling", ["max", "avg"]) == "max":
            if pooling[i] == "max":

                x =  layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool'+str(j+1))(x)
            else:
                x =  layers.AveragePooling2D(pool_size=(3,3), strides=(2,2), name='maxpool'+str(j+1))(x)


            j = j+2
            filter_size = filter_size+16
            
        num_fire2 = hp.Int('num_fire_2',0,2, default=2)
        use_bypass2 = [ hp.Boolean('use_bypass_2'+str(i)) for i in range(num_fire2)]
         
        print(use_bypass2)
         
        for i in range(num_fire2): 
            x = create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j))
            x = create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j+1),use_bypass=use_bypass2[i])
            filter_size = filter_size+16
            j = j+2

            #x = create_fire_module(x, int(filter_size*compression_val), name='fire8')
            #x = create_fire_module(x, int(filter_size*compression_val), name='fire9',use_bypass=hp.Boolean('use_bypass'))
        
        
        
        dropout_rate = hp.Float('dropout_rate',0.0,0.8)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = output(x, self.classes)

        model = keras.Model(inputs=input_img, outputs=x)

        optimizer = hp.Choice("optimizer", ["RMSprop", "sgd"])
        model.compile(
            optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
    
class SqueezeNetSEAutoModel(HyperModel):
    def __init__(self, classes, IMG_WIDTH, IMG_HEIGHT):
        self.classes = classes
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def get_axis(self):
        axis = -1 if K.image_data_format() == 'channels_last' else 1
        return axis

    def create_fire_module(self,x, nb_squeeze_filter, name, use_bypass=False):
        """
        Creates a fire module
        
        Arguments:
            x                 : input
            nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
            use_bypass        : if True then a bypass will be added
            name              : name of module e.g. fire123
        
        Returns:
            x                 : returns a fire module
        """
      
        nb_expand_filter = 4 * nb_squeeze_filter
        squeeze    = layers.Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
        expand_1x1 = layers.Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
        expand_3x3 = layers.Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
        
        axis = self.get_axis()
        x_ret =  layers.Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
        
        if use_bypass:
            x_ret =  layers.Add(name='%s_concatenate_bypass'%name)([x_ret, x])
            
        return x_ret


    def output(self,x, nb_classes):
        x = layers.Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
        x = layers.GlobalAveragePooling2D(name='avgpool10')(x)
        x = layers.Activation("softmax", name='softmax')(x)
        return x

    def squeeze_excitation_layer(self,input_layer, out_dim, ratio, conv=False):
    
        squeeze =  layers.GlobalAveragePooling2D()(input_layer)

        excitation =  layers.Dense(units=out_dim / ratio, activation='relu')(squeeze)
        excitation =  layers.Dense(out_dim,activation='sigmoid')(excitation)
        excitation =  tf.reshape(excitation, [-1,1,1,out_dim])

        scale =  layers.multiply([input_layer, excitation])

        if conv:
            shortcut = layers.Conv2D(out_dim,kernel_size=1,strides=1,
                                            padding='same',kernel_initializer='he_normal')(input_layer)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = input_layer
            
        out = scale # tf.keras.layers.add([shortcut, scale])
        return out

    def build(self, hp):
       
          
        compression_val = hp.Fixed('compression',1.0)
    
        
        input_img =  layers.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))

        
        x = layers.Conv2D(int(96*compression_val), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
         

        j = 2
        filter_size = 16

        num_fire = hp.Int("fire_module", 1,2, default=2)
        use_bypass = [ hp.Boolean('use_bypass'+str(i)) for i in range(num_fire)]
        pooling = [ hp.Choice('pooling'+str(i), ["max", "avg"]) for i in range(num_fire) ]
        
        print(use_bypass)
        print(pooling)
        for i in range(num_fire):
         
            ratio = hp.Choice("squeeze_ratio"+str(j),[8,16,32], default=32)
            x_in = x
            x = self.create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j), )
            
            se_insert = hp.Boolean("SE_add"+str(j))
            
            if se_insert:
                x = self.squeeze_excitation_layer(x, out_dim=x.shape[3], ratio=ratio)
                
                if hp.Boolean("SE_skip"+str(j)):
                    if x_in.shape[3] != x.shape[3] :
                        x_in =  layers.Conv2D(x.shape[3],kernel_size=1,strides=1,
                                      padding='same',kernel_initializer='he_normal')(x_in)
                        x_in = layers.BatchNormalization()(x_in)
                        
                    x = layers.add([x_in, x])
                                     
            
            ratio = hp.Choice("squeeze_ratio"+str(j),[8,16,32], default=32)
            x_in = x
            x = self.create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j+1),use_bypass=use_bypass[i])

            se_insert = hp.Boolean("SE_add"+str(j+1))
            
            if se_insert:
                 
                x = self.squeeze_excitation_layer(x, out_dim=x.shape[3], ratio=ratio)
                
                if hp.Boolean("SE_skip"+str(j+1)):
                    if x_in.shape[3] != x.shape[3] :
                        x_in =  layers.Conv2D(x.shape[3],kernel_size=1,strides=1,
                                      padding='same',kernel_initializer='he_normal')(x_in)
                        x_in =layers.BatchNormalization()(x_in)
                        
                    x = layers.add([x_in, x])
                     
            
                
            #if hp.Choice("pooling", ["max", "avg"]) == "max":
            if pooling[i] == "max":

                x =  layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool'+str(j+1))(x)
            else:
                x =  layers.AveragePooling2D(pool_size=(3,3), strides=(2,2), name='avgpool'+str(j+1))(x)


            j = j+2
            filter_size = filter_size+16
            
        num_fire2 = hp.Int('num_fire_2',0,2, default=2)
        use_bypass2 = [ hp.Boolean('use_bypass_2'+str(i)) for i in range(num_fire2)]
         
        print(use_bypass2)
         
        for i in range(num_fire2): 
            
            ratio = hp.Choice("squeeze_ratio"+str(j),[8,16,32], default=32)
            
            x_in = x
            x = self.create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j))
            
            se_insert = hp.Boolean("SE_add"+str(j))
            
            if se_insert:
                x = self.squeeze_excitation_layer(x, out_dim=x.shape[3], ratio=ratio)
                if hp.Boolean("SE_skip"+str(j)):
                    if x_in.shape[3] != x.shape[3] :
                        x_in =  layers.Conv2D(x.shape[3],kernel_size=1,strides=1,
                                          padding='same',kernel_initializer='he_normal')(x_in)
                        x_in = layers.BatchNormalization()(x_in)
                    x = layers.add([x_in, x])
                
            ratio = hp.Choice("squeeze_ratio"+str(j),[8,16,32], default=32)
            
            x_in = x
            x = self.create_fire_module(x, int(filter_size*compression_val), name='fire'+str(j+1),use_bypass=use_bypass2[i])
            
            se_insert = hp.Boolean("SE_add"+str(j+1))
            
            if se_insert:
                x = self.squeeze_excitation_layer(x, out_dim=x.shape[3], ratio=ratio)
                if hp.Boolean("SE_skip"+str(j+1)):
                    if x_in.shape[3] != x.shape[3] :
                        x_in =  layers.Conv2D(x.shape[3],kernel_size=1,strides=1,
                                          padding='same',kernel_initializer='he_normal')(x_in)
                        x_in = layers.BatchNormalization()(x_in)
                    x = layers.add([x_in, x])
                    
            filter_size = filter_size+16
            j = j+2

            #x = create_fire_module(x, int(filter_size*compression_val), name='fire8')
            #x = create_fire_module(x, int(filter_size*compression_val), name='fire9',use_bypass=hp.Boolean('use_bypass'))
        
        
        
        dropout_rate = hp.Float('dropout_rate',0.0,0.8)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = self.output(x, self.classes)

        model = keras.Model(inputs=input_img, outputs=x)

        optimizer = hp.Choice("optimizer", ["RMSprop", "sgd"])
        model.compile(
            optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
    
