#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#----------------
# imports
#---------------
import tensorflow as tf
import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2 
import math
from glob import glob
from tqdm.auto import tqdm
from scipy.special import softmax
from .utils import LOG_INFO,padWords


#----------------
# model
#---------------
class DotAttention(tf.keras.layers.Layer):
    """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output
    """
    def __init__(self):
        super().__init__()
        self.inf_val=-1e9
        
    def call(self,q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
       
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * self.inf_val)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output

class RobustScanner(object):
    def __init__(self,model_dir):
        
        #-------------
        # config-globals
        #-------------
        with open(os.path.join(model_dir,"rec","vocab.json")) as f:
            vocab = json.load(f)["vocab"]    
        self.vocab=vocab

        #-------------------
        # fixed params
        #------------------
        self.img_height  =  64
        self.img_width   =  512
        self.nb_channels =  1
        self.pos_max     =  40          
        self.enc_filters =  256
        self.factor      =  32

        # calculated
        self.enc_shape   =(self.img_height//self.factor,self.img_width//self.factor, self.enc_filters )
        self.attn_shape  =(None, self.enc_filters )
        self.mask_len    =int((self.img_width//self.factor)*(self.img_height//self.factor))
        
        self.start_end=len(vocab)+1
        self.pad_value=len(vocab)+2

        LOG_INFO(f"Label len:{self.pos_max}")
        LOG_INFO(f"Vocab len:{len(self.vocab)}")
        LOG_INFO(f"Pad Value:{self.pad_value}")
        LOG_INFO(f"Start End:{self.start_end}")

        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        with strategy.scope():
            self.encm    =  self.encoder()
            self.encm.load_weights(os.path.join(model_dir,"rec","enc.h5"))      
            LOG_INFO("encm loaded")
            self.seqm    =  self.seq_decoder()
            self.seqm.load_weights(os.path.join(model_dir,"rec","seq.h5"))      
            LOG_INFO("seqm loaded")
            
            self.posm    =  self.pos_decoder()
            self.posm.load_weights(os.path.join(model_dir,"rec","pos.h5"))      
            LOG_INFO("posm loaded")
            
            self.fusm    =  self.fusion()
            self.fusm.load_weights(os.path.join(model_dir,"rec","fuse.h5"))      
            LOG_INFO("fusm loaded")
        
    def encoder(self):
        '''
        creates the encoder part:
        * defatult backbone : DenseNet121 **changeable
        args:
        img           : input image layer
            
        returns:
        enc           : channel reduced feature layer

        '''
        # img input
        img=tf.keras.Input(shape=(self.img_height,self.img_width,self.nb_channels),name='image')
        # backbone
        backbone=tf.keras.applications.DenseNet121(input_tensor=img ,weights=None,include_top=False)
        # feat_out
        enc=backbone.output
        # enc 
        enc=tf.keras.layers.Conv2D(self.enc_filters,kernel_size=3,padding="same")(enc)
        return tf.keras.Model(inputs=img,outputs=enc,name="rs_encoder")

    def seq_decoder(self):
        '''
        sequence attention decoder (for training)
        Tensorflow implementation of : 
        https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/sequence_attention_decoder.py
        '''
        # label input
        gt=tf.keras.Input(shape=(self.pos_max,),dtype='int32',name="label")
        # mask
        mask=tf.keras.Input(shape=(self.pos_max,self.mask_len),dtype='float32',name="mask")
        # encoder
        enc=tf.keras.Input(shape=self.enc_shape,name='enc_seq')
        
        # embedding,weights=[seq_emb_weight]
        embedding=tf.keras.layers.Embedding(self.pad_value+2,self.enc_filters)(gt)
        # sequence layer (2xlstm)
        lstm=tf.keras.layers.LSTM(self.enc_filters,return_sequences=True)(embedding)
        query=tf.keras.layers.LSTM(self.enc_filters,return_sequences=True)(lstm)
        # attention modeling
        # value
        bs,h,w,nc=enc.shape
        value=tf.keras.layers.Reshape((h*w,nc))(enc)
        attn=DotAttention()(query,value,value,mask)
        return tf.keras.Model(inputs=[gt,enc,mask],outputs=attn,name="rs_seq_decoder")
    


    def pos_decoder(self):
        '''
        position attention decoder (for training)
        Tensorflow implementation of : 
        https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/position_attention_decoder.py
        '''
        # pos input
        pt=tf.keras.Input(shape=(self.pos_max,),dtype='int32',name="pos")
        # mask
        mask=tf.keras.Input(shape=(self.pos_max,self.mask_len),dtype='float32',name="mask")
        # encoder
        enc=tf.keras.Input(shape=self.enc_shape,name='enc_pos')
        
        # embedding,weights=[pos_emb_weight]
        query=tf.keras.layers.Embedding(self.pos_max+1,self.enc_filters)(pt)
        # part-1:position_aware_module
        bs,h,w,nc=enc.shape
        value=tf.keras.layers.Reshape((h*w,nc))(enc)
        # sequence layer (2xlstm)
        lstm=tf.keras.layers.LSTM(self.enc_filters,return_sequences=True)(value)
        x=tf.keras.layers.LSTM(self.enc_filters,return_sequences=True)(lstm)
        x=tf.keras.layers.Reshape((h,w,nc))(x)
        # mixer
        x=tf.keras.layers.Conv2D(self.enc_filters,kernel_size=3,padding="same")(x)
        x=tf.keras.layers.Activation("relu")(x)
        key=tf.keras.layers.Conv2D(self.enc_filters,kernel_size=3,padding="same")(x)
        bs,h,w,c=key.shape
        key=tf.keras.layers.Reshape((h*w,nc))(key)
        attn=DotAttention()(query,key,value,mask)
        return tf.keras.Model(inputs=[pt,enc,mask],outputs=attn,name="rs_pos_decoder")

    def fusion(self):
        '''
        fuse the output of gt_attn and pt_attn 
        '''
        # label input
        gt_attn=tf.keras.Input(shape=self.attn_shape,name="gt_attn")
        # pos input
        pt_attn=tf.keras.Input(shape=self.attn_shape,name="pt_attn")
        
        x=tf.keras.layers.Concatenate()([gt_attn,pt_attn])
        # Linear
        x=tf.keras.layers.Dense(self.enc_filters*2,activation=None)(x)
        # GLU
        xl=tf.keras.layers.Activation("linear")(x)
        xs=tf.keras.layers.Activation("sigmoid")(x)
        x =tf.keras.layers.Multiply()([xl,xs])
        # prediction
        x=tf.keras.layers.Dense(self.pad_value+1,activation=None)(x)
        return tf.keras.Model(inputs=[gt_attn,pt_attn],outputs=x,name="rs_fusion")


        
    def porcess_data(self,crops):
        images=[]
        masks=[]
        poss=[]
            
        for crop in tqdm(crops):
            # word
            word,vmask=padWords(crop,(self.img_height,self.img_width),ptype="left")
            word=cv2.cvtColor(word,cv2.COLOR_BGR2GRAY)
            plt.imshow(word)
            plt.show()
            word=np.expand_dims(word,axis=-1) 
            word=np.expand_dims(word,axis=0) 
            # image
            images.append(word)
            # mask
            vmask=math.ceil((vmask/self.img_width)*(self.img_width//self.factor))
            mask_dim=(self.img_height//self.factor,self.img_width//self.factor)
            imask=np.zeros(mask_dim)
            imask[:,:vmask]=1
            imask=imask.flatten().tolist()
            imask=[1-int(i) for i in imask]
            imask=np.stack([imask for _ in range(self.pos_max)])
            masks.append(imask)
            # pos
            pos=[i for i in np.arange(0,self.pos_max)]
            poss.append(pos)  
        
        return images,masks,poss

    def predict_on_batch(self,batch,infer_len):
        '''
            predicts on batch
        '''
        # process batch data
        image=batch["image"]
        label=batch["label"]
        pos  =batch["pos"]
        mask =batch["mask"]
        # feat
        enc=self.encm.predict(image)
        pt_attn=self.posm.predict({"pos":pos,"enc_pos":enc,"mask":mask})
        for i in range(infer_len):
            gt_attn=self.seqm.predict({"label":label,"enc_seq":enc,"mask":mask})
            step_gt_attn=gt_attn[:,i,:]
            step_pt_attn=pt_attn[:,i,:]
            pred=self.fusm.predict({"gt_attn":step_gt_attn,"pt_attn":step_pt_attn})
            char_out=softmax(pred,axis=-1)
            max_idx =np.argmax(char_out,axis=-1)
            if i < self.pos_max - 1:
                label[:, i + 1] = max_idx
        texts=[]
        for w_label in label:
            _label=[]
            for v in w_label[1:]:
                if v==self.start_end:
                    break
                _label.append(v)
            texts.append("".join([self.vocab[l] for l in _label]))
        return texts

    def recognize(self,crops,batch_size=32,infer_len=10):
        '''
            final wrapper
        '''
        texts=[]
        images,masks,poss=self.porcess_data(crops)
        
        for idx in tqdm(range(0,len(images),batch_size)):
            batch={}
            # image
            batch["image"]=images[idx:idx+batch_size]
            batch["image"]=np.vstack(batch["image"])
            batch["image"]=batch["image"]/255.0
            # pos
            batch["pos"]  =poss[idx:idx+batch_size]
            batch["pos"]  =np.vstack(batch["pos"])
            # mask
            batch["mask"]  =[np.expand_dims(mask,axis=0) for mask in masks[idx:idx+batch_size]]
            batch["mask"]  =np.vstack(batch["mask"])
            # label
            batch["label"] =np.ones_like(batch["pos"])*self.start_end
            # recog
            texts+=self.predict_on_batch(batch,infer_len)
        return texts