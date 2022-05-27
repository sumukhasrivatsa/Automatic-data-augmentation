import NETA
from NETA import netA
import NETC1
from NETC1 import netC
import torch.nn as nn
import torch.optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import netA1
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class trainer_class(nn.Module):
    def __init__(self,train_loader,validation_loader,args):
        super().__init__()
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        
        self.args=args


    def train(self):
        ##do not write outer loop from here 
        val_loader=iter(self.validation_loader)

        ##define outer loop from here 
        ###get images from train loader
        for i,(images,labels) in enumerate(self.train_loader):
            

            
            ##create an object of netA and send image batch
            neta=netA1.netA()

            ##defining an optimizer for netA
            opt_neta=torch.optim.Adam(neta.fc_loc.parameters(),lr=1)


            rot_imgs,rot_labels,affine_matrix,inverse_matrix=neta.transform_images(images,labels)
            
            
            ##^ obtained the rotated images from netA
            ##now we have both images and rot_imgs
            
            ##object of netc
            netc=NETC1.netC()


            ##defining an optimizer for netc
            opt_netc=torch.optim.Adam(netc.parameters(),lr=self.args.learning_rate)

            ##passing rotated images through netc

            output=netc.forward(rot_imgs,labels)


            ##defining loss
            loss_obj=nn.CrossEntropyLoss()
            loss=loss_obj(output,labels)
            
            ##gradients before backward for neta
            #for param in neta.fc_loc.parameters():
            #    print(param.data)

            ##gradients before backward for netc
            #for param in netc.parameters():
            #   print(param.data)
            
            #print(neta.fc_loc[0].weight.grad)
            #print(netc.conv1.weight.grad)


            loss.backward()


            #print(neta.fc_loc[0].weight.grad)
            ##UPDATE only netc
            opt_netc.step()

            opt_netc.zero_grad()
            opt_neta.zero_grad()    



            ##print(netc.conv1.weight.grad)


            
            ##get the next from the validation batch
            val_imgs,val_labels=val_loader.next()
            
            
           
            img=val_imgs[0][0,:,:].numpy()
            plt.imshow(img)
            plt.show()
            
            # apply transformation +ve
            grids1 = F.affine_grid(affine_matrix, val_imgs.size(), align_corners=True)
            rot_val_imgs = F.grid_sample(val_imgs,grids1, align_corners=True)
            img=rot_val_imgs[0][0,:,:].detach().numpy()
            plt.imshow(img)
            plt.show()
            

            ##apply inverse transformation
            grids2 = F.affine_grid(inverse_matrix, val_imgs.size(), align_corners=True)
            rot_val_imgs_inv = F.grid_sample(rot_val_imgs,grids2, align_corners=True)
            img=rot_val_imgs_inv[0][0,:,:].detach().numpy()
            plt.imshow(img)
            plt.show()


            outputs2=netc.forward(rot_val_imgs_inv,val_labels)
            val_loss=loss_obj(outputs2,val_labels)



            opt_neta.zero_grad()
            opt_netc.zero_grad()


            val_loss.backward()
            opt_neta.step()
        

            print("************")

            ##updating only netc
            

            ##neta gradients after backward
            #for param in neta.fc_loc.parameters():
            #    print(param.data)
            print("****************************************************************")
            ##netc gradients after backward

            #for param in netc.parameters():
             #   print(param.data)
            
            #rots_images=rot_imgs[5,0,:,:]
            #rots_images=rots_images.detach().numpy()
            
            #rots_images=rot_imgs[0,0,:,:]
            #rots_images=rots_images.detach().numpy()
            #print(rots_images.shape)


            ###thing is i need to use the same grid with same netA 
           

            if i==3:
                break

        ###pass them to get the noise and the transformed images
        ## pass the transformed and regular images to the NETC
        ##use the loss to only back propogate through the netc and not neta
        
