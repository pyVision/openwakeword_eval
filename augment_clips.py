
import openwakeword.data
from scipy.io.wavfile import write

import os
import collections
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from tqdm import tqdm
import openwakeword
import openwakeword.data
import openwakeword.utils
import openwakeword.metrics
import torchaudio
import os
import argparse
import torch
import random
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import acoustics
# from pydub import AudioSegment
# from pydub.playback import play

from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.signal_processing import reverberate

#python augment_speech.py --negative_path datasets/MS-SNSD/noise_train/ --positive_path aout1 --output_path augmented1/ --snr_low -5 --snr_high 10

class AugmentSpeech():


    def add_reverb(self,samples_path,rir_path,output_path,batch_size=8,combined_size=2,reverb_factor=1.0):


        j = 0
        #print("adding reverb")
        sample_clips, sample_durations = openwakeword.data.filter_audio_paths(
            [
                samples_path
            ],
            duration_method = "header", # use the file header to calculate duration
            min_length_secs=0,
            max_length_secs=60*30
        )

        rirs, rirs_durations = openwakeword.data.filter_audio_paths(
            [
                rir_path
            ],
            duration_method = "header", # use the file header to calculate duration
            min_length_secs=0,
            max_length_secs=60*30
        )

        mixing_generator=self.__add_reverb__(sample_clips,rirs,output_path,batch_size,combined_size,reverb_factor)
        
        N_total=len(sample_clips)

        for batch in tqdm(mixing_generator, total=N_total//batch_size):
            #batch = batch[0]
            for i in range(len(batch)):
                print("saving file",sample_clips[j+i],batch[i].shape,(j+i),len(batch[i]))
                write(output_path+"/"+Path(sample_clips[j+i]).stem+".wav", 16000, batch[i].astype(np.int16))

            j=j+len(batch)

        #mixed_clips=next(mixing_generator)
        print("completed with speech augmentation",len(sample_clips))





    def __add_reverb__(self,sample_clips,rirs,output_path,batch_size=8,combined_size=2,reverb_factor=1.0,pre_delay=0.1,dry_wet_ratio=0.5):

        
 

        start_index = [0]*batch_size
        labels = [0]*len(sample_clips)

        #print("lables ",labels)

       

        sr = 16000
        total_length_seconds = combined_size # must be the some window length as that used for the negative examples
        total_length = int(sr*total_length_seconds)
        
        for i in range(0, len(sample_clips), batch_size):


            # Load foreground clips/start indices and truncate as needed
            sr = 16000
            start_index_batch = start_index[i:i+batch_size]
            foreground_clips_batch = [read_audio(j) for j in sample_clips[i:i+batch_size]]
            foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]

            labels_batch = np.array(labels[i:i+batch_size])

            mixed_clips = []
            for r in foreground_clips_batch:
                 combined_clip = torch.zeros(total_length)
                 start=0
                 # Apply pre-delay if specified
                 if pre_delay > 0:
                    pre_delay_samples = int(pre_delay * 16000)  # Assuming 16kHz sample rate
                    pre_delay_padding = torch.zeros(pre_delay_samples)
                    audio_tensor = torch.cat((pre_delay_padding, r), 0)
                 else:
                    audio_tensor=r

                 if total_length >audio_tensor.shape[0]:
                    eindex=audio_tensor.shape[0]
                 else:
                    eindex=total_length

                 combined_clip[start:start + eindex] = combined_clip[start:start + eindex] + audio_tensor[:eindex]

                 mixed_clips.append(combined_clip)
    

            mixed_clips_batch = torch.vstack(mixed_clips)
            
            rir_waveform, sr = torchaudio.load(random.choice(rirs))
            if rir_waveform.shape[0] > 1:
                rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
            scaled_rir = rir_waveform * reverb_factor
            rir_waveform=scaled_rir



            mixed_clips_batch1 = reverberate(mixed_clips_batch, rir_waveform, rescale_amp="avg")

            mixed_clips_batch = dry_wet_ratio * mixed_clips_batch1 + (1 - dry_wet_ratio) * mixed_clips_batch

            # Normalize clips only if max value is outside of [-1, 1]
            abs_max, _ = torch.max(
                torch.abs(mixed_clips_batch), dim=1, keepdim=True
            )
            mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

            # Convert to 16-bit PCM audio
            mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)

            # Remove any clips that are silent (happens rarely when mixing/reverberating)
            #error_index = torch.from_numpy(np.where(np.max(mixed_clips_batch,axis=1) != 0)[0])
            #print("error index is ",error_index)        


            #mixed_clips_batch = mixed_clips_batch[error_index]

            yield (mixed_clips_batch)
            



    def __change_speed__(self,audio_batch, sample_rate, speed_factor):
        """
        Change the speed of a batch of audio tensors without changing their pitch.
        
        Parameters:
        audio_batch (torch.Tensor): The batch of audio tensors of shape (batch_size, num_channels, num_samples).
        sample_rate (int): The sample rate of the audio.
        speed_factor (float): The factor by which to change the speed (e.g., 0.8 for slower, 1.2 for faster).

        Returns:
        torch.Tensor: The speed-adjusted batch of audio tensors.
        """
        # Calculate the new sample rate
        new_sample_rate = int(sample_rate * speed_factor)
        
        # Initialize the resampler
        print("resamples",sample_rate,new_sample_rate)
        resample = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        
        # Apply the resampling to the batch
        audio_resampled = resample(audio_batch)
        print("resamples")
        
        return audio_resampled


    def change_speed(self,samples_path,output_path,speed_factors,total_length,batch_size):


        positive_clips, durations = openwakeword.data.filter_audio_paths(
            [
                samples_path
            ],
            min_length_secs = 0.4, # minimum clip length in seconds
            max_length_secs = 10, # maximum clip length in seconds
            duration_method = "header" # use the file header to calculate duration
        )
        sr=16000
        total_length_seconds = 2 # must be the some window length as that used for the negative examples
        total_length = int(sr*total_length_seconds)
        j=0
        for i in range(0, len(positive_clips), batch_size):
        
            
            foreground_clips_batch = [read_audio(j) for j in positive_clips[i:i+batch_size]]
            foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]
            start=0
            speed_batch = np.random.choice(speed_factors)
            mixed_clips=[]
            k=0
            for audio_tensor in foreground_clips_batch:
                print("pre processing clip",positive_clips[i+k],audio_tensor.shape[0],durations[i+k],i+k)
                combined_clip = torch.zeros(total_length)
                
                if total_length >audio_tensor.shape[0]:
                    eindex=audio_tensor.shape[0]
                else:
                    eindex=total_length
                combined_clip[start:start + eindex] = combined_clip[start:start + eindex] + audio_tensor[:eindex]

                #transformed_audio = self.__change_speed__(combined_clip,sr,factor)
 
                mixed_clips.append(combined_clip[0:total_length])
                k=k+1

            
            mixed_clips_batch = torch.vstack(mixed_clips)

            abs_max, _ = torch.max(
                torch.abs(mixed_clips_batch), dim=1, keepdim=True
            )
            mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

            mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)   

            audio = (librosa.effects.time_stretch(mixed_clips_batch/32767.0, rate=speed_batch)*32767).astype(np.int16)

            N_total=len(mixed_clips_batch)     

            for j in range(N_total):
                write(output_path+"/"+Path(positive_clips[j+i]).stem+"_"+str(speed_batch)+".wav", 16000, audio[j])  

            #j=j+1 



    def __add_colored_noise__(self,sample_clips,output_path,snr_low, snr_high,batch_size=8,combined_size=2):

        
 

        start_index = [0]*batch_size
        labels = [0]*len(sample_clips)

        #print("lables ",labels)

        snrs_db = np.random.uniform(snr_low, snr_high, batch_size)

        sr = 16000
        total_length_seconds = combined_size # must be the some window length as that used for the negative examples
        total_length = int(sr*total_length_seconds)
        
        for i in range(0, len(sample_clips), batch_size):


            # Load foreground clips/start indices and truncate as needed
            sr = 16000
            start_index_batch = start_index[i:i+batch_size]
            foreground_clips_batch = [read_audio(j) for j in sample_clips[i:i+batch_size]]
            foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]

            labels_batch = np.array(labels[i:i+batch_size])

            mixed_clips = []
            for r in foreground_clips_batch:
                 combined_clip = torch.zeros(total_length)
                 start=0
                 # Apply pre-delay if specified

                 audio_tensor=r

                 if total_length >audio_tensor.shape[0]:
                    eindex=audio_tensor.shape[0]
                 else:
                    eindex=total_length

                 combined_clip[start:start + eindex] = combined_clip[start:start + eindex] + audio_tensor[:eindex]

                 noise_color = ["white", "pink", "blue", "brown", "violet"]
                 noise_clip = acoustics.generator.noise(total_length, color=np.random.choice(noise_color))
                 noise_clip = torch.from_numpy(noise_clip/noise_clip.max())
                 mixed_clip = openwakeword.data.mix_clip(combined_clip, noise_clip, np.random.choice(snrs_db), 0)

                 mixed_clips.append(combined_clip)
    

            mixed_clips_batch = torch.vstack(mixed_clips)

            # Normalize clips only if max value is outside of [-1, 1]
            abs_max, _ = torch.max(
                torch.abs(mixed_clips_batch), dim=1, keepdim=True
            )
            mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

            # Convert to 16-bit PCM audio
            mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)

            # Remove any clips that are silent (happens rarely when mixing/reverberating)
            #error_index = torch.from_numpy(np.where(np.max(mixed_clips_batch,axis=1) != 0)[0])
            #print("error index is ",error_index)        


            #mixed_clips_batch = mixed_clips_batch[error_index]

            yield (mixed_clips_batch)


    def add_colored_noise(self,samples_path,output_path,snr_low, snr_high,batch_size):

                
        j = 0
        #print("adding reverb")
        sample_clips, sample_durations = openwakeword.data.filter_audio_paths(
            [
                samples_path
            ],
            duration_method = "header", # use the file header to calculate duration
            min_length_secs=0,
            max_length_secs=60*30
        )



        mixing_generator=self.__add_colored_noise__(sample_clips,output_path,snr_low, snr_high,batch_size,2)
        
        
        N_total=len(sample_clips)

        for batch in tqdm(mixing_generator, total=N_total//batch_size):
            #batch = batch[0]
            for i in range(len(batch)):
                print("saving file",sample_clips[j+i],batch[i].shape,(j+i),len(batch[i]))
                write(output_path+"/"+Path(sample_clips[j+i]).stem+".wav", 16000, batch[i].astype(np.int16))

            j=j+len(batch)

        #mixed_clips=next(mixing_generator)
        print("completed with speech augmentation",len(sample_clips))


        





    def add_noise(self,negative_path,positive_path,output_path,snr_low,snr_high,batch_size):

        negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
            [
                negative_path
            ],
            min_length_secs = 1, # minimum clip length in seconds
            max_length_secs = 60*30, # maximum clip length in seconds
            duration_method = "header" # use the file header to calculate duration
        )


        print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(negative_durations)//3600} hours")


        positive_clips, durations = openwakeword.data.filter_audio_paths(
            [
                positive_path
            ],
            min_length_secs = 0.4, # minimum clip length in seconds
            max_length_secs = 10, # maximum clip length in seconds
            duration_method = "header" # use the file header to calculate duration
        )

        print(f"{len(positive_clips)} positive clips after filtering")




        # Define starting point for each positive clip based on its length, so that each one ends 
        # between 0-200 ms from the end of the total window size chosen for the model.
        # This results in the model being most confident in the prediction right after the
        # end of the wakeword in the audio stream, reducing latency in operation.

        # Get start and end positions for the positive audio in the full window
        sr = 16000
        total_length_seconds = 2 # must be the some window length as that used for the negative examples
        total_length = int(sr*total_length_seconds)

        jitters = (np.random.uniform(0, 0.2, len(positive_clips))*sr).astype(np.int32)
        starts = [total_length - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]
        #print(starts)
        starts = [num if num >=0 else 0 for num in starts]
        #print(starts)
        ends = [int(i*sr) + j for i, j in zip(durations, starts)]

        N_total = len(positive_clips) # maximum number of rows in mmap file


        



        print("parameters are ",total_length,batch_size)
        mixing_generator = openwakeword.data.mix_clips_batch(
            foreground_clips = positive_clips,
            background_clips = negative_clips,
            combined_size = total_length,
            batch_size = batch_size,
            #rirs = rir_clips,
            #rir_probability = 1,
            snr_low = snr_low,
            snr_high = snr_high,
            start_index = starts,
            shuffle = False,
            volume_augmentation=False, # randomly scale the volume of the audio after mixing
          
            #return_background_clips_delay = (16000, 24000),
        )

        
        # In[ ]:
        j = 0
        

        for batch in tqdm(mixing_generator, total=N_total//batch_size):
            batch, lbls, background = batch[0], batch[1], batch[2]
            for i in range(len(batch)):
                write(output_path+"/"+Path(positive_clips[j+i]).stem+".wav", 16000, batch[i].astype(np.int16))

            j=j+len(batch)


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='augment noise,reverb,speed to audio files')
    parser.add_argument('--negative_path', type=str, required=False, help='Path to the noise files')
    parser.add_argument('--samples_path', type=str, required=True, help='Path to the clean audio files')
    parser.add_argument('--output_path', type=str, required=False, help='Path to save the augmented files')
    parser.add_argument('--snr_low', type=float, required=False, help='Lower bound of SNR')
    parser.add_argument('--snr_high', type=float, required=False, help='Upper bound of SNR')
    parser.add_argument('--type', type=str, required=True, help='NOISE,REVERB,SPEED,COLORED_NOISE')
    parser.add_argument('--batch_size', type=str, required=False, help='batch size')
    parser.add_argument('--rir_path', type=str, required=False, help=' RIR path')
    parser.add_argument('--speed_factors', type=float, nargs='+', required=False, help='List of speed factors to apply.')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    if args.type=="NOISE":
        a = AugmentSpeech()
        a.add_noise(
            negative_path=args.negative_path,
            samples_path=args.samples_path,
            output_path=args.output_path,
            snr_low=args.snr_low,
            snr_high=args.snr_high,
            batch_size=int(args.batch_size)
        )
    elif args.type=="REVERB":
        print("starting with speech augmentation")
        a = AugmentSpeech()
        a.add_reverb(
            samples_path=args.samples_path,
            rir_path=args.rir_path,
            output_path=args.output_path,
            batch_size=int(args.batch_size),
            reverb_factor=10
        )       
    elif args.type=="SPEED":
        print("starting with SPEED augmentation")
        a = AugmentSpeech()
        a.change_speed(
            samples_path=args.samples_path,
            speed_factors=args.speed_factors,
            output_path=args.output_path,
            total_length=2,
            batch_size=int(args.batch_size)
            
        )    
    elif args.type=="COLORED_NOISE":
        print("starting with COLORED_NOISE augmentation")
        a = AugmentSpeech()
        a.add_colored_noise(
            samples_path=args.samples_path,
            output_path=args.output_path,
            snr_low=args.snr_low,
            snr_high=args.snr_high,
            batch_size=int(args.batch_size)
        )  

    # a = AugmentSpeech()
    # a.plot_reverb("aout1/c12723a065264eee9ac1d6fe93eb4bf7.wav","reverb1/c12723a065264eee9ac1d6fe93eb4bf7.wav")
# a=AugmentSpeec
# a.add_noise(negative_path="datasets/MS-SNSD/noise_train/",positive_path="aout1",output_path="augmented1/",snr_low=-5,snr_high=10)



# %%
""