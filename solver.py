import torch
from dataloader import CrackDataset
from torchvision import transforms as T
import torch.utils.data as data
import random
import pdb
from model import Model
from PIL import Image
import os
import numpy as np
from datetime import datetime
class Solver():

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def _calc_precision(self,probs, labels):
        _, pred = probs.max(dim=1) # <- class. 0:Background, 1:crack
        tp = (pred[labels == 1] == 1).float() # producing error when ....mean() is empty 
        fp = (pred[labels == 0] == 1).float()
        if not len(tp) == 0:
            tp = tp.mean()
        if not len(fp) == 0:
            fp = fp.mean()
        return tp/(fp + tp + 1e-6)

    def _calc_recall(self, probs, labels):
        _, pred = probs.max(dim=1) # <- class. 0:Background, 1:crack
        tp = (pred[labels == 1] == 1).float()
        fn = (pred[labels == 1] == 0).float()
        if not len(tp) == 0:
            tp = tp.mean()
        if not len(fn) == 0:
            fn = fn.mean()
        return tp/(fn + tp + 1e-6)

    def _calc_f1_score(self,probs, labels):

        prec = self._calc_precision(probs, labels)
        recall = self._calc_recall(probs, labels)
        f1 = 2 * (prec * recall)/(recall + prec + 1e-6) # 1e-6 to limit nan errors
        return f1

    def _build_model(self):
        self.model = Model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)
    
    def _build_transform(self):
        transforms = []
        # transforms.append(T.Resize(self.config.img_size))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transforms = T.Compose(transforms)
        
    def _tile_image(self, img):
        """ Breaks image into (img_size, img_size). Assumes PIL input type.
            Discards regions at the corners that are less than img_size. 
            One could potentially resize the corner cases and add them to the list.
            ----> Rasterizes row wise!
            Returns a Tensor object"""
        
        # img = np.array(img)
        tiled_imgs = []
        hei, wid = img.size
        tile_img_size = self.config.img_size

        # Number of cropped images across hei and wid
        no_tiles_hei = hei // tile_img_size
        no_tiles_wid = wid // tile_img_size

        for i in range(no_tiles_hei):
            for j in range(no_tiles_wid):
                # cropped_img = img[i * tile_img_size: (i + 1) * tile_img_size,
                #                   j * tile_img_size: (j + 1) * tile_img_size, :]
                cropped_img = img.crop((j * tile_img_size,        i * tile_img_size,
                                        (j + 1) * tile_img_size, (i + 1) * tile_img_size))
                # cropped_img = np.transpose(cropped_img, (2, 0, 1))
                cropped_img_T = self.transforms(cropped_img).unsqueeze(0)
                tiled_imgs.append(cropped_img_T)
        
        return torch.cat(tiled_imgs, dim=0).to(self.device)
    
    def train(self):
        
        # Dataloader and transform function
        # Can also make img_size greater than model input and include random crop into the transform -> Can possibly generalize better
        self._build_transform()

        # Model and loss
        self._build_model()
        ce_loss = torch.nn.CrossEntropyLoss().to(self.device) 
        softmax = torch.nn.Softmax(dim=1)
        min_f1_score = 0
        
        
        dataset = CrackDataset(self.config.img_dir, transform = self.transforms)
        data_loader = data.DataLoader(dataset, shuffle = True, 
                                      num_workers = self.config.num_workers, 
                                      batch_size = self.config.batch_size)
    
        for epoch in range(self.config.epochs):
            print("Running epoch - {}".format(epoch))
            
            # Run over training set
            train_f1_score = 0
            data_loader.dataset.make_train()
            self.model.train()
            for i, (imgs, labels) in enumerate(data_loader):
                logits = self.model(imgs)
                loss = ce_loss(logits, labels.squeeze().long())
                loss.backward()
                self.optimizer.step()
                
                probs = softmax(logits)
                # try:
                train_f1_score += self._calc_f1_score(probs, labels.squeeze())  
                # except:
                # import pdb;pdb.set_trace()        
            train_f1_score = train_f1_score/len(data_loader)

            # Run over validation set
            val_f1_score = 0
            data_loader.dataset.make_eval()
            self.model.eval()
            for i, (imgs, labels) in enumerate(data_loader):
                logits = self.model(imgs)
                probs = softmax(logits)
                val_f1_score +=self._calc_f1_score(probs, labels.squeeze())

            val_f1_score = val_f1_score/len(data_loader)

            # Save model based on f1-score
            print("Current epoch: \n train_f1_score: {},   val_f1_score: {}".format(train_f1_score, val_f1_score))
            if val_f1_score > min_f1_score:
                min_f1_score = val_f1_score
                print("Saving model for {} epoch: val_f1_score = {} ---".format(epoch, val_f1_score))
                torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, '{0}-epoch-{1:.4f}.pt'.format(epoch, val_f1_score)))


    def test(self):
        """ Breaks all the images in self.config.test_dir and runs them through the trained ConvNet.
            Superimposes the output onto each loaded image and saves the output image into an 'out + time()' folder.
        """

        # Create transform
        self._build_transform()

        output_save_dir = os.path.join(self.config.test_dir, 'out_' + datetime.today().strftime('%m-%d--%H:%M'))
        if not os.path.isdir(output_save_dir):
            os.makedirs(output_save_dir)
        
        # Load saved model
        self._build_model()
        self.model.load_state_dict(torch.load(self.config.saved_model))
        self.model.eval()
        softmax = torch.nn.Softmax(dim=1)
        
        # Create tiles of input images (sequentially) -> Preserves memory usage
        # Get only files in self.config.test_dir (discard directories in test dir)
        imgs_file_names = [f for f in os.listdir(self.config.test_dir) if os.path.isfile(os.path.join(self.config.test_dir, f))]
        for file_no, img_file_name in enumerate(imgs_file_names):
            img = Image.open(os.path.join(self.config.test_dir, img_file_name))
            
            # Image can be reconstructed      
            img_hei, img_wid = img.size
            no_tiles_wid = img_wid // self.config.img_size
            no_tiles_hei = img_hei // self.config.img_size
            
            # Tile image into smaller images of (img_size x img_size)
            tiled_imgs = self._tile_image(img)

            # Store predictions of ConvNet in output. Not bool because can be extended to multiple class labels
            output = np.zeros(no_tiles_hei * no_tiles_wid)
            
            # Pass through ConvNet
            no_of_batches = len(tiled_imgs)//self.config.batch_size
            
            for i in range(no_of_batches):
                curr_batch = tiled_imgs[i * self.config.batch_size: (i + 1) * self.config.batch_size]
                out = self.model(curr_batch)
                probs = softmax(out)
                _, preds = probs.max(dim=1)
                
                output[i * self.config.batch_size:(i + 1) * self.config.batch_size] = preds.cpu().numpy()

            # Pass the remaining images through the convNet
            last_batch = tiled_imgs[no_of_batches * self.config.batch_size:len(tiled_imgs)]
            out = self.model(last_batch)
            probs = softmax(out)
            _, preds = probs.max(dim=1)
            output[no_of_batches * self.config.batch_size:len(tiled_imgs)] = preds.cpu().numpy()
            
            # Superimpose mask onto img and save
            output_img = np.zeros((3, no_tiles_hei * self.config.img_size, no_tiles_wid * self.config.img_size))
            tile_img_size = self.config.img_size
            

            j = -1
            for i, img in enumerate(tiled_imgs):
                curr_i = i % no_tiles_wid
                if curr_i == 0:
                    j += 1

                # print("Curr j: {}, curr_i: {}, curr_output: {}".format(j, curr_i, curr_i + j* no_tiles_wid)) 
                # denormalize and write image only if pred==1 followed by denormalizing

                output_img[:, j * tile_img_size:(j + 1) * tile_img_size, 
                           curr_i * tile_img_size:(curr_i + 1) * tile_img_size]  = \
                                img.mul_(0.5).add_(0.5).cpu().numpy() * output[curr_i + j* no_tiles_wid]    
            
            # Save output image
            print("Opened {}x{} image and saving {}x{} image with {}x{} hei x wid tiles".format(img_hei, img_wid, no_tiles_hei * tile_img_size, no_tiles_wid * tile_img_size, no_tiles_hei, no_tiles_hei ))
            output_img = np.transpose(output_img, (1, 2, 0))
            im = Image.fromarray(np.uint8(output_img * 255))
            im.save(os.path.join(output_save_dir, 'out_{}.jpg'.format(file_no)))
            

            
            