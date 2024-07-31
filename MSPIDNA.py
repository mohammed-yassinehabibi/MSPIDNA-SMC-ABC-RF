from tqdm import tqdm
import psutil
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, Subset, DataLoader



class MSPIDNASimulation:
    def __init__(self, idx=0, nb_iter=1, data_folder = None, 
                 results_folder = None, device=None, dataset_name=f"dataset_0_prior_1_adaptative.pt", 
                 model_folder=None):
        self.model_folder = model_folder if model_folder is not None else ''
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model = False
        self.parallelize = True
        self.training = True
        self.last_model = False
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.idx = idx
        self.nb_iter = nb_iter
        self.dataset_name = dataset_name
        self.dataset = self.load_and_normalize_dataset()
        self.num_output = self.dataset[0][1].shape[0]
        self.params_model = {
            'load_model': self.load_model,
            'model_path': self.model_folder + f'trained_model.pth',
            'num_output': self.num_output,
            'num_block': 1,
            'num_feature': self.num_output+3,
            'nb_classes': 16,
            'device': self.device
        }
        self.model_to_test = self.choose_model()
        self.params_training = {
            'model': self.model_to_test,
            'criterion': nn.SmoothL1Loss(reduction='mean', beta=0.3).to(self.device),
            'optimizer': optim.Adam(self.model_to_test.parameters(), lr=1e-3),
            'dataset': Subset(self.dataset, list(range(len(self.dataset)))),
            'device': self.device,
            'num_epochs': 5,
            'batch_size': 16,
            'save': True,
            'best_loss_path': self.model_folder + f'best_loss.pth',
            'best_model_path': self.model_folder + f'trained_model.pth',
            'last_model': self.last_model
        }
        self.params_eval = {
            'model': self.model_to_test,
            'dataset': self.dataset,
            'device': self.device,
            'to_predict': [
                'mutation_rate', 
                'transition_matrix_AT', 'transition_matrix_AC', 'transition_matrix_AG',
                'transition_matrix_TA', 'transition_matrix_TC', 'transition_matrix_TG',
                'transition_matrix_CA', 'transition_matrix_CT', 'transition_matrix_CG',
                'transition_matrix_GA', 'transition_matrix_GT', 'transition_matrix_GC'
            ] if self.num_output==13 else ['mutation_rate', 
                                           'transition_matrix_AA', 'transition_matrix_AT', 'transition_matrix_AC', 'transition_matrix_AG',
                                            'transition_matrix_TA', 'transition_matrix_TT', 'transition_matrix_TC', 'transition_matrix_TG',
                                            'transition_matrix_CA', 'transition_matrix_CT', 'transition_matrix_CC', 'transition_matrix_CG',
                                            'transition_matrix_GA', 'transition_matrix_GT', 'transition_matrix_GC', 'transition_matrix_GG'
            ]
        }

    def load_and_normalize_dataset(self):
        print("LOADING X,y...")
        dataset = torch.load(self.data_folder + self.dataset_name, weights_only=False)
        print("LOADED")
        print(f"Memory utilization: {psutil.virtual_memory().percent}%")
        print("Train_test_split")
        len_train = 4 * len(dataset) // 5
        y = dataset[:][1]
        y_train = y[:len_train]
        y_train_mean = y_train.mean(axis=0)
        y_train_std = y_train.std(axis=0)
        torch.save(y_train_mean, self.model_folder + f'y_train_mean.pt')
        torch.save(y_train_std, self.model_folder + f'y_train_std.pt')
        y_norm = (y - y_train_mean) / y_train_std
        print("Create new normalised dataset...")
        dataset2 = TensorDataset(dataset[:][0], y_norm)
        dataset = None
        print(f"Memory utilization: {psutil.virtual_memory().percent}%")
        return dataset2

    def choose_model(self):
        model = self.get_model(**self.params_model)
        if self.parallelize and not self.load_model:
            if torch.cuda.device_count() > 1:
                print(f"Utilisation de {torch.cuda.device_count()} GPU.")
                model = nn.DataParallel(model)
        if isinstance(model, nn.DataParallel) and self.load_model and not self.parallelize:
            model = model.module.to(self.device)
        else:
            model = model.to(self.device)
        return model

    def get_model(self, **params_model):
        if params_model['load_model']:
            model = torch.load(params_model['model_path'], map_location=params_model['device'], weights_only=False)
            if isinstance(model, nn.DataParallel):
                model.module.device = params_model['device']
            else:
                model.device = params_model['device']
        else:
            model = MSPIDNA(**params_model)
        return model

    def train_model(self):
        train(**self.params_training)

    def eval_model(self):
        return eval(**self.params_eval)

    def run(self):
        if self.training:
            os.makedirs(os.path.dirname(self.results_folder), exist_ok=True)
            self.train_model()
        predictions = self.eval_model()
        y_train_mean = torch.load(self.model_folder + f'y_train_mean.pt', weights_only=False)
        y_train_std = torch.load(self.model_folder + f'y_train_std.pt', weights_only=False)
        norm_predictions = predictions * y_train_std.to('cpu').repeat(1, 2).numpy() + y_train_mean.to('cpu').repeat(1, 2).numpy()
        print(f"CPU utilization: {psutil.cpu_percent()}%")
        print(f"Memory utilization: {psutil.virtual_memory().percent}%")
        print('mse : ', mse(norm_predictions))

class MSPIDNABlock(nn.Module):
    def __init__(self, num_output, num_feature):
        super(MSPIDNABlock, self).__init__()
        self.num_output = num_output
        self.phi = nn.Conv2d(num_feature * 2, num_feature*2, (1, 3))
        self.phi_bn = nn.BatchNorm2d(num_feature * 2)
        self.maxpool = nn.MaxPool2d((1, 2))
        self.fc = nn.Linear(num_feature*2, num_output)
        
    def forward(self, x, output):
        x = self.phi(self.phi_bn(x))
        psi1 = torch.mean(x, 2, keepdim=True)
        psi = psi1
        current_output = self.fc(torch.mean(psi[:, :, :, :], 3).squeeze(2))
        output = output + current_output
        psi = psi.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, psi), 1)
        x = F.relu(self.maxpool(x))
        return x, output

class MSPIDNA(nn.Module):
    '''
    MSPIDNA architecture
    '''
    def __init__(self, num_output, num_block, num_feature, nb_classes, device, **kwargs):
        super(MSPIDNA, self).__init__()
        self.num_output = num_output
        self.conv_pos = nn.Conv3d(1, num_feature, (1, 3, nb_classes), padding=(0,1,0))
        self.conv_pos_bn = nn.BatchNorm3d(num_feature)
        self.conv_snp = nn.Conv3d(1, num_feature, (1, 3, nb_classes), padding=(0,1,0))
        self.conv_snp_bn = nn.BatchNorm3d(num_feature)
        self.blocks = nn.ModuleList([MSPIDNABlock(num_output, num_feature) for i in range(num_block)])

        self.device = device

    def forward(self, x):
        pos = x[:, 0, :, :].view(x.shape[0], 1, 1, x.shape[2], -1)
        snp = x[:, 1:, :, :].unsqueeze(1)
        pos = F.relu(self.conv_pos_bn(self.conv_pos(pos))).expand(-1, -1, snp.size(2), -1, -1)
        snp = F.relu(self.conv_snp_bn(self.conv_snp(snp)))
        x = torch.cat((pos, snp), 1).squeeze(-1)
        output = torch.zeros(x.size(0), self.num_output).to(self.device)
        for block in self.blocks:
            x, output = block(x, output)
        return output.squeeze(1)  # Squeeze to make it a 1-dimensional output

# Training loop
def train(model,optimizer,criterion,device,dataset=None, num_epochs=10,batch_size=128,save=False,last_model=False,best_loss_path='results/best_loss.pth',best_model_path='results/trained_model.pth'):
    print("TRAINING")
    if dataset==None :
        return "No dataset !!"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) #target.unsqueeze(0) or not ? cf warning
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')
        
        if save:  #Save the best model with best loss and update it if needed
            if os.path.exists(best_loss_path):
                if running_loss/ len(dataloader) < torch.load(best_loss_path, weights_only=False):
                    torch.save(running_loss/ len(dataloader), best_loss_path)
                    torch.save(model, best_model_path)
                    print('Model and best_loss updated')
            else:
                torch.save(running_loss/ len(dataloader), best_loss_path)
                torch.save(model, best_model_path)
                print('Model and best_loss updated')
            if last_model and epoch == num_epochs-1:
                torch.save(model, best_model_path)
                print('Last model saved')

def eval(model, device, dataset=None, to_predict = None):
    print("EVALUATING")
    model.eval()
    predictions = pd.DataFrame()
    dataloader = DataLoader(dataset, batch_size=16)
    with torch.no_grad():
        all_output = torch.Tensor()
        true_rates = torch.Tensor()
        for input, target in tqdm(dataloader, desc="Evaluating"):
            input = input.float().to(device)
            output = model(input).to('cpu') #CPU to be able to convert to DataFrame
            all_output = torch.cat((all_output, output), 0)
            true_rates = torch.cat((true_rates, target.to('cpu')), 0)
        pred = pd.DataFrame(all_output.numpy())
        pred = pred.rename(columns={i: ('pred_' + to_predict[i]) for i in range(len(to_predict))})
        true_rates = pd.DataFrame(true_rates.numpy(), columns=to_predict)
        predictions = pd.concat([true_rates, pred], axis = 1)
    return predictions

class GenerateSummaryStatistics:
    '''
    Generate summary statistics for the points to infer with a trained MSPIDNA
    Inputs:
    - nb_iter: number of iterations T
    - idx: index of the point to infer
    - data_folder: folder containing the data points
    - results_folder: folder to save the summary statistics
    - goal: boolean to generate summary statistics for the points to infer
    - model_folder: folder containing the trained MSPIDNA
    '''
    def __init__(self,nb_iter, idx, data_folder, results_folder, goal = False, model_folder=None):
        self.nb_iter = nb_iter
        self.idx = idx
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.goal = goal
        self.model_folder = model_folder if model_folder is not None else ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params_model = {
            'model_path': self.model_folder + f'trained_model.pth',
            'device': self.device
        }
        self.params_to_predict = [
                'mutation_rate', 
                'transition_matrix_AT', 'transition_matrix_AC', 'transition_matrix_AG',
                'transition_matrix_TA', 'transition_matrix_TC', 'transition_matrix_TG',
                'transition_matrix_CA', 'transition_matrix_CT', 'transition_matrix_CG',
                'transition_matrix_GA', 'transition_matrix_GT', 'transition_matrix_GC'
            ] if 'null_diag' in model_folder else ['mutation_rate', 
                                           'transition_matrix_AA', 'transition_matrix_AT', 'transition_matrix_AC', 'transition_matrix_AG',
                                            'transition_matrix_TA', 'transition_matrix_TT', 'transition_matrix_TC', 'transition_matrix_TG',
                                            'transition_matrix_CA', 'transition_matrix_CT', 'transition_matrix_CC', 'transition_matrix_CG',
                                            'transition_matrix_GA', 'transition_matrix_GT', 'transition_matrix_GC', 'transition_matrix_GG'
            ]
        
    #Model to use to generate summary-statistics
    def model(self):
        model = torch.load(self.params_model['model_path'], map_location=self.params_model['device'], weights_only=False)
        if isinstance(model, torch.nn.DataParallel):
            model.module.device = self.params_model['device']
            model = model.module.to(self.device) #to de-parallelize the model
        else:
            model = model.to(self.device)
        return model
    
    def run(self):
        #Load model and data
        model_to_test = self.model()
        if self.goal and "real" in self.data_folder:
            if "full_seq_null_diag" in self.data_folder:
                dataset = torch.load('mtDNA/full_seq_null_diag_real_data/test_dataset_mtDNA_L4.pt', weights_only=False)
            elif "full_seq" in self.data_folder:
                dataset = torch.load('mtDNA/full_seq_real_data/test_dataset_mtDNA_L4.pt', weights_only=False)
            elif "null_diag" in self.data_folder:
                dataset = torch.load('mtDNA/null_diag_real_data/test_dataset_mtDNA_L4.pt', weights_only=False)
            else:
                dataset = torch.load('mtDNA/real_data/test_dataset_mtDNA_L4.pt', weights_only=False)
        elif self.goal:
            dataset = torch.load(self.data_folder+f"dataset_{self.idx}_prior_{self.nb_iter}_goals.pt", weights_only=False)
        else:
            dataset = torch.load(self.data_folder+f"dataset_{self.idx}_prior_{self.nb_iter}.pt", weights_only=False)
        #Normalize data
        y_train_mean = torch.load(self.model_folder+f'y_train_mean.pt', weights_only=False)
        y_train_std = torch.load(self.model_folder+f'y_train_std.pt', weights_only=False)
        y = dataset[:][1]
        y_norm = (y - y_train_mean)/y_train_std
        dataset2 = TensorDataset(dataset[:][0], y_norm)
        dataset = None
        #Generate summary-statistics
        summary_statistics = eval(model_to_test, self.device, dataset2, self.params_to_predict)
        norm_summary_statistics = summary_statistics*y_train_std.to('cpu').repeat(1,2).numpy()+y_train_mean.to('cpu').repeat(1,2).numpy()
        if self.goal:
            torch.save(norm_summary_statistics,self.results_folder+f'parameters_and_SS_{self.idx}_prior_{self.nb_iter}_goals.pt')
        else:
            torch.save(norm_summary_statistics,self.results_folder+f'parameters_and_SS_{self.idx}_prior_{self.nb_iter}.pt')
        print('mse : ',mse(norm_summary_statistics))

def mse(predictions):
    mid = predictions.shape[1]//2
    return mean_squared_error(predictions.iloc[:,:mid],predictions.iloc[:,mid:])

