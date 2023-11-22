# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from import_data_utils import *

# internal imports for RDC
sys.path.insert(1, '../../')
from utils.imagenet_dataset import get_dataset
from utils.image_processing import preprocess_image_drc
from models.student_models  import student3
from AlexNet_keras_model import model
# -

res_path = "/home/mor/NDS_project/results_data/sub"


# +
# # allocating memory
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[3], 'GPU')
# -

class pic_edit():
    """
    This class should ease applying filters and edits to the cut squered pictue before they are being proccessed by the net. 
    It aims to contain all present and future potential dynamic properties of a picture.
    """
    def __init__(self, size = 224, blur = 1):
        self.size = size
        self.blur = blur
        
    def __repr__(self):
        return f'pic({self.size, self.blur})'
    
    def __hash__(self):
        return hash((self.size, self.blur))


# +
# Some functions

# image resizing and padding
def resize_padding(x, des_size):
    """
    This function takes the image which is already in the desired size, but minimizes it to a smaller size
    and add padding to keep the original size.  
    """
    if x.shape[0] < des_size:
        raise ValueError("Desired size must be smaller than original size!")
    neutral_color = x[0,0,0]  #pixel from the image surrounding
    padding = np.ones(x.shape) * neutral_color
    start =(x.shape[0] - des_size)//2
    end = start + des_size
    resized_im = keras.layers.Resizing(des_size, des_size, interpolation='bilinear', crop_to_aspect_ratio=False,)(x).numpy()
    padding[start:end, start:end] = resized_im
    return padding


# -

class Subject:
    """
    A subject instance will hold together all the network's data for the networks that had been trained over his images and 
    were compared to his brain activity.
    """
    def __init__(self, num:int, path:str):
        self.num = num
        self.images, self.voxels =  import_data_for_sub(self.num, path)
        self.nets = {}
        self.batch_size = 64
        self.net_lim = (len( self.images) // self.batch_size) * self.batch_size
        self.RDM_fMRI = self.calc_RDM()
        
    def calc_RDM(self):
        RDM_fmri = {}
        for area in self.voxels.keys():
            RDM_fmri[area] = {}
            for roi in self.voxels[area].keys():
                RDM_fmri[area][roi] = 1-np.corrcoef(self.voxels[area][roi][:self.net_lim,:])
            print("Created RDM for: " + str(area))
        return RDM_fmri
    
    def save_subject(self, path):
        run_time = datetime.now().strftime("%Y%m%d%H%M")
        path_for_save = path + str(self.num) + "/results_" + run_time
        # Check whether the specified path exists or not
        if not os.path.exists(path_for_save):
           # Create a new directory because it does not exist
           os.makedirs(path_for_save)
           print("The new directory "+ path_for_save +" is created!")
        
        with open(path_for_save + '/sub1_data.pkl', 'wb') as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)
            print("saved corr_dict!")
        
    @classmethod
    def load_sub_pickle(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


# +
def preprocess_for_drc_image(x,**kwargs):
    low_res_frames, _ = preprocess_image_drc(x, x.shape[-2], kwargs['res'], is_training=False,teacher_mode=False,**kwargs)
    return low_res_frames


def preprocess_for_drc_batch(x, **kwargs):
    s = np.shape(x)
    n_steps = kwargs['n_steps']
    res = kwargs['res']
    xout = np.zeros([s[0],n_steps,res,res,s[-1]])
    for ii, this_image in enumerate(x):
        xout[ii] = preprocess_for_drc_image(tf.convert_to_tensor(this_image,dtype=tf.float32),**kwargs)
    return xout

class Network:
    def __init__(self, net_type:str, net_arch:str, amp=0):
        """
        initialization
        net_type : {FF / DRC}
        net_arch : {Alexnet / Vgg}
        net : keras object
        """
        self.net_type_name =  net_type
        self.net_type = FeedForward() if net_type == 'FF' else DRC()
        self.net_arch = VGG(net_type, amp) if net_arch == 'Vgg' else AlexNet(net_type, amp)
        self.input_size = self.net_arch.net_size
            
class FeedForward:
    # CNN applying
    def preds_from_batches(self, layer_to_eval, data,batch_size = 256, sample_size=-1, drc_args=None ):
        """
        Sasha's function for CNN application. Processing each layer and saving the output
        """
        data_len = data.shape[0]
        samples_set_flag = False
        for batch in range(data_len//batch_size):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, data_len)
            result = layer_to_eval(data[start:end]).numpy().reshape([end-start,-1])
            if not samples_set_flag and sample_size > 0:
                samples = np.random.choice(result.shape[-1], sample_size)
                samples_set_flag = True
            if batch == 0:
                ll_out = np.zeros([data_len,result.shape[-1]]) if sample_size == -1 else np.zeros([data_len,sample_size])
            ll_out[start:end,:] = result if sample_size == -1 else result[:,samples]
        return ll_out
    
    def create_RDM(self, RDM_obj):
        RDM_net_sz={}
        for ll, layer in enumerate(RDM_obj.net.net_arch.net.layers):
                print(ll, layer.name)
                feature_extractor = keras.Model(inputs=RDM_obj.net.net_arch.net.inputs,outputs=layer.output,)
                activity = self.preds_from_batches(feature_extractor, RDM_obj.input_net, batch_size=64, sample_size=3000)
                RDM_net_sz[ll] = 1-np.corrcoef(activity)
                del feature_extractor
        return RDM_net_sz
    

    
class DRC:
    def preds_from_batches(self, layer_to_eval, data, batch_size = 64, sample_size=-1, time_mode='mean', drc_args={}):
        """
        Sasha's function for CNN application with a DRC layer. Processing each layer and saving the output
        """
        data_len = data.shape[0]
        samples_set_flag=False

        for batch in range(data_len//batch_size):
            start = batch * batch_size
            end = min((batch + 1)*batch_size, data_len)
            result = layer_to_eval(preprocess_for_drc_batch(data[start:end], **drc_args)) # <------
            result = result.numpy()
            if np.ndim(result) == 5:
                if time_mode == 'mean':
                    result = result.mean(axis=1)
                else:
                    error
            result = result.reshape([end-start,-1])
            if not samples_set_flag and sample_size > 0:
                samples = np.random.choice(result.shape[-1],sample_size)
                samples_set_flag = True
            if batch == 0:
                ll_out = np.zeros([data_len, result.shape[-1]]) if sample_size == -1 else np.zeros([data_len, sample_size])
            ll_out[start:end, :] = result if sample_size == -1 else result[:, samples]
        return ll_out

    
    def create_RDM(self, RDM_obj):
        RDM_net_sz={}
        for ll,layer in enumerate(RDM_obj.net.net_arch.student.layers + RDM_obj.net.net_arch.decoder2.layers): # layers of 2 networks
            print(ll, layer.name)

            # This is where we build an appropriate feature_extractor for each layer according to the net part it belongs with
            if ll<len(RDM_obj.net.net_arch.student.layers): # apply student
                feature_extractor = keras.Model(
                   inputs=RDM_obj.net.net_arch.student.inputs,
                   outputs=layer.output,
                )
            else: # apply decoder on the results of student
                decoder_model = keras.Model(inputs=RDM_obj.net.net_arch.decoder2.inputs, outputs = layer.output)
                input0 = keras.layers.Input(shape=(5, 56, 56, 3))
                stu_output = RDM_obj.net.net_arch.student(input0)
                dec_output = decoder_model(stu_output)
                net = keras.models.Model(inputs=[input0], outputs=dec_output, name='frontend')
                feature_extractor = keras.Model(
                inputs=input0,
                outputs=dec_output,
                )
            # here we apply the net's prediction of the particular layer
            params = RDM_obj.pic_params
            input_net = inputs=RDM_obj.input_net # <- most probably resizing to 224*224
            if ll == 0:
                print("applying network for images with params: \r\n image size:" + str(params.size) + 
                  " || blur filter size: "+ str(params.blur) + "\r\n amp frame: " + str(RDM_obj.net.net_arch.drc_args['amp']))
            activity = self.preds_from_batches(feature_extractor, input_net, # <---------
                                          batch_size=64,
                                          sample_size=3000,
                                          time_mode='mean',
                                          drc_args=RDM_obj.net.net_arch.drc_args)
            RDM_net_sz[ll] = 1-np.corrcoef(activity)
            del feature_extractor
        return RDM_net_sz

        
class AlexNet:
    def __init__(self, net_type, amp=0):
        self.net_size = 227
        if net_type == "FF":
            self.net = ...  # Initialize net for AlexNet-FeedForward
        elif net_type == "DRC":
            self.net = ...  # Initialize net for AlexNet-DRC

class VGG:
    def __init__(self, net_type, amp=0):
        self.net_size = 224
        if net_type == "FF":
            self.net = keras.applications.VGG16(input_shape=(net_size, net_size, 3),
                                include_top=True, weights='imagenet')  # Initialize net for VGG-FeedForward
        elif net_type == "DRC":
            self.drc_args=dict(preprocessing='default',rggb_mode=False,
            return_position_info=False, offsets = None,
            unprocess_high_res=False,enable_random_gains=False,n_steps=5,res=56,
            central_squeeze_and_pad_factor=-1, amp=amp)

            drc_fe_args = {'sample': 5, 'res': 56, 'activation': 'relu', 'dropout': 0.0, 
                           'rnn_dropout': 0.0, 'num_features': 128, 'rnn_layer1': 32, 'rnn_layer2': 64, 
                           'layer_norm': False, 'batch_norm': False, 'conv_rnn_type': 'lstm', 
                           'block_size': 1, 'add_coordinates': 0, 'time_pool': 'average_pool', 'dense_interface': True,
                           'loss': 'mean_squared_error', 'upsample': 0, 'pos_det': None, 'enable_inputB': False,
                           'expanded_inputB': False, 'rggb_ext_type': 0, 'channels': 3, 'updwn': 1,
                           'kernel_size': 3, 'custom_metrics': []}

            parameters = {}

            parameters['pretrained_student_path'] = '/home/arivkind/pretrained_models/vgg16drc_baseline/noname_j1131336_t1662126871_feature_net_ckpt'
            parameters['pretrained_decoder_path'] = '/home/arivkind/pretrained_models/vgg16drc_baseline/noname_j1131336_t1662126871_saved_models/decoder_trained_model_fix0/'
            self.student = student3(**drc_fe_args)
            load_status1 = self.student.load_weights(parameters['pretrained_student_path'])
            self.decoder2 = keras.models.load_model(parameters['pretrained_decoder_path'])
            net_input = keras.layers.Input(shape=(5, 56, 56, 3))
            net_output = self.decoder2(self.student(net_input))
            self.net = keras.models.Model(inputs=[net_input], outputs=net_output, name='frontend')  # Initialize net for VGG-DRC


# -

class RDM:
    def __init__(self, sub, net, pic_params):
        self.pic_params = pic_params
        self.net = net
        self.input_net = self.preprocss_net(sub)
        self.RDM = self.create_RDM()
        
    def preprocss_net(self, sub):
        print('starting preprocessing')
        """
        This function exectute all needed and optional steps before implementing the network over the data
        it resizes the input, and when needed it decreses and creates padding
        """ 
        cut_ims = sub.images
        input_len = len(cut_ims)
        input_net = np.zeros((input_len, self.net.input_size, self.net.input_size,3))
        for i in range(input_len):       
            input_net[i] = keras.layers.Resizing(self.net.input_size, self.net.input_size, interpolation='bilinear', crop_to_aspect_ratio=False,)(cut_ims[i]).numpy()
            if self.pic_params.blur > 0:
                input_net[i] = cv2.blur(input_net[i], (self.pic_params.blur, self.pic_params.blur))
            if self.pic_params.size < self.net.input_size:
                input_net[i] = resize_padding(input_net[i], self.pic_params.size)
            elif self.pic_params.size > self.net.input_size:
                cut = (pic_params.size-self.net.input_size)//2
                zoomin = keras.layers.Resizing(self.pic_params.size, self.pic_params.size, interpolation='bilinear', crop_to_aspect_ratio=False,)(input_net[i]).numpy()
                cutzoom = zoomin[cut:self.net.input_size+cut,cut:self.net.input_size+cut]
                if i < 1:
                    plt.imshow(cutzoom.astype(int), vmin=0, vmax=255)
                    plt.show()
                input_net[i] = cutzoom
        input_net = input_net[:sub.net_lim,...]
        print(self.net.net_type_name)
        if self.net.net_type_name == 'DRC':
            print('DRC')
            input_net = input_net * (1/256)
            return input_net
        else:
            print('FF')
            input_net = input_net.astype(int)
            return keras.applications.vgg16.preprocess_input(input_net)
    
    def create_RDM(self):
        return self.net.net_type.create_RDM(self)
    
    def get_RDM(self):
        return (self.RDM)
# +
# Functions for calculating the RDM between network and 
def resample(mat,mat_):
    ii = np.random.randint(mat.shape[0],size=(mat.shape[0]))
    not_diag = np.eye(*mat.shape)<1e-5
    return mat[:,ii][ii,:],mat_[:,ii][ii,:],not_diag[:,ii][ii,:]


def corr_list_extractor_resampled(RDM_net, RDM_fmri,corr_fun=spearmanr):
    """
    fOR BOOTSTRAP This function takes all RDMs from all network layers, and all RDMS from FMRI voxels and calculate correlations.
    """
    corr_dict = {}
    for brain_area in RDM_fmri.keys():
        corr_dict[brain_area] = []
        for cnn_layer in range(len(RDM_net)):
            sz=RDM_net[cnn_layer].shape[0]
            mask = np.triu(np.ones([sz,sz],dtype=np.bool),1)
            RDM1, RDM2, resample_mask = resample(RDM_net[cnn_layer],RDM_fmri[brain_area])
            mask = np.logical_and(mask, resample_mask)
            corr, pval = corr_fun(np.nan_to_num(RDM1[mask]),
                                  RDM2[mask])
            corr_dict[brain_area].append(corr)
        corr_dict[brain_area] = np.array(corr_dict[brain_area])#.reshape([2,-1])
    print ("done correlations ")
    return corr_dict

def corr_list_extractor(RDM_net, RDM_human, corr_fun=spearmanr):
    """
    This function takes all RDMs from all network layers, and all RDMS from FMRI voxels and calculate correlations.
    """
    corr_dict = {}
    for brain_area in RDM_human.keys():
        corr_dict[brain_area] = []
        for cnn_layer in range(len(RDM_net)):
            sz=RDM_net[cnn_layer].shape[0]
            mask = np.triu(np.ones([sz,sz],dtype=np.bool),1)
            corr, pval = corr_fun(np.nan_to_num(RDM_net[cnn_layer][mask]), RDM_human[brain_area][mask])
            corr_dict[brain_area].append(corr)
    print ("done correlations ")
    return corr_dict



# +
# Functions for calculating the RDM between network and 
def resample(mat,mat_):
    ii = np.random.randint(mat.shape[0],size=(mat.shape[0]))
    not_diag = np.eye(*mat.shape)<1e-5
    return mat[:,ii][ii,:],mat_[:,ii][ii,:],not_diag[:,ii][ii,:]

def corr_extractor_resample(RDM_net, RDM_human, corr_fun=spearmanr, n=10):
    """
    This function takes RDM from a network layer, and RDM from an human data voxels and calculate correlations.
    """
    corlist = np.zeros(n)
    for i in range(n):
        sz=RDM_net.shape[0]
        mask = np.triu(np.ones([sz,sz],dtype=np.bool),1)
        RDM1, RDM2, resample_mask = resample(RDM_net,RDM_human)
        mask = np.logical_and(mask, resample_mask)
        corlist[i], pval = corr_fun(np.nan_to_num(RDM1[mask]), RDM2[mask])
    return {'mean': np.mean(corlist), 'sd': np.std(corlist)}

def corr_list_extractor_resampled_recursive(RDM_net, RDM_human, corr_fun=spearmanr):
    """
    Process the data which can either be a dictionary or an ndarray.
    """
    cor_dict = {}
    if isinstance(RDM_human, dict):
        for key_human, value_human in RDM_human.items():
            if isinstance(value_human, dict):
                # If the value is another dictionary, call the function recursively
                cor_dict[key_human] = corr_list_extractor_resampled_recursive(RDM_net, value_human)
            
            elif isinstance(value_human, np.ndarray):
                # Process the ndarray
                print(f"Processing ndarray for key: {key_human}")
                if isinstance(RDM_net, dict):
                    cor_dict[key_human] = {}
                    for key_net, value_net in RDM_net.items():
                        if isinstance(value_net, dict):
                            cor_dict[key_human][key_net] = corr_list_extractor_resampled_recursive(value_net, value_human)
                        elif isinstance(value_net, np.ndarray):
                            cor_dict[key_human][key_net] = corr_extractor_resample(value_net, value_human)
                elif isinstance(RDM_net, np.ndarray):
                    cor_dict[key_human] = corr_extractor_resample(RDM_net, value_human)
                # Add your ndarray processing code here
            else:
                print(f"Unrecognized type for key: {key_human}")
    elif isinstance(RDM_human, np.ndarray):
        if isinstance(RDM_net, dict):
            for key_net, value_net in RDM_net.items():
                if isinstance(value_net, dict):
                    cor_dict[key_net] = corr_list_extractor_resampled_recursive(value_net, RDM_human)
                elif isinstance(key_net, np.ndarray):
                    cor_dict[key_net] = corr_extractor_resample(value_net, RDM_human)
        elif isinstance(RDM_net, np.ndarray):
            return corr_extractor_resample(RDM_net, RDM_human)
                    
        # Process the ndarray directly
        print("Processing ndarray directly.")
        # Add your ndarray processing code here
    else:
        print("Unrecognized type.")
    return cor_dict


# -

sub1 = Subject(1, path)

net_size = 224
pic_params = pic_edit(net_size,0)
VGGNet1 = Network('FF', 'Vgg')
sub1_vgg_RDM = RDM(sub1, VGGNet1, pic_params)


# +
DRCNet1 = Network('DRC', 'Vgg', 4)
sub1_vgg_RDM_DRC = RDM(sub1, DRCNet1, pic_params)

# DRCNet1.net_arch.__dict__
# -

n_resamples = 2
corr_dict_vis = {samp: corr_list_extractor_resampled(sub1_vgg_RDM.get_RDM(), sub1.RDM_fMRI['prf-visualrois'])  for samp in range(n_resamples)}

corr_dict_vis


def corr_list_extractor_resampled1(RDM_net,RDM_fmri,corr_fun=spearmanr):
    """
    fOR BOOTSTRAP This function takes all RDMs from all network layers, and all RDMS from FMRI voxels and calculate correlations.
    """
    corr_dict = {}
    for brain_area in RDM_fmri.keys():
        corr_dict[brain_area] = []
        for cnn_layer in range(len(RDM_net)):
            sz=RDM_net[cnn_layer].shape[0]
            mask = np.triu(np.ones([sz,sz],dtype=np.bool),1)
            RDM1, RDM2, resample_mask = resample(RDM_net[cnn_layer],RDM_fmri[brain_area])
            mask = np.logical_and(mask, resample_mask)
            corr, pval = corr_fun(np.nan_to_num(RDM1[mask]),
                                  RDM2[mask])
            corr_dict[brain_area].append(corr)
        corr_dict[brain_area] = np.array(corr_dict[brain_area])#.reshape([2,-1])
    print ("done correlations ")
    return corr_dict


corr_dict_vis_DRC = {samp: corr_list_extractor_resampled(sub1_vgg_RDM_DRC.get_RDM(), sub1.RDM_fMRI['prf-visualrois'])  for samp in range(1)}


corr_dict_vis_DRC

sub1.RDM_fMRI['prf-visualrois']
sub1_vgg_RDM.get_RDM()

cor_dict = corr_list_extractor_resampled_recursive(sub1_vgg_RDM.get_RDM(), sub1.RDM_fMRI['prf-visualrois'], corr_fun=spearmanr)

cor_dict_drc = corr_list_extractor_resampled_recursive(sub1_vgg_RDM_DRC.get_RDM(), sub1.RDM_fMRI['prf-visualrois'], corr_fun=spearmanr)

cor_dict

cor_dict_drc


