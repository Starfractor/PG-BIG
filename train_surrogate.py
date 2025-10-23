import os
import torch
import torch.nn as nn
from dataset.dataset_183_retarget import retargeted183_data_loader
from tqdm import tqdm
from write_mot import write_muscle_activations

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        
        # Define MLP layers
        self.fc_main = nn.Linear(input_dim, output_dim, bias=False)


        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # Flatten the input (assuming x is of shape [batch_size, 196, 33])
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, 196*33]

        main_out = self.fc_main(x)

        # Pass through the network
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)  # Output is [batch_size, 196*80]
        
        # x = main_out + x
        x = main_out

        # Reshape output to [batch_size, 196, 80]
        x = x.contiguous().view(batch_size, -1, 80)
        x = torch.tanh(x)/2 + 0.5
        return x
    
window_size = 64
batch_size = 16
# Use the retargeted 183 dataset loader
train_loader = retargeted183_data_loader(window_size=window_size,
                                         unit_length=4,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         data_dir='/home/mnt/datasets/183_retargeted',
                                         pre_load=False)

# For simplicity treat this surrogate training as a reconstruction task
# where inputs == targets. Infer input/output shapes from one batch.
sample_batch = next(iter(train_loader))
# sample_batch: (motion, len_motion, name, subject_id)
motion_sample = sample_batch[0]
# motion_sample shape: (batch_size, T, D)
_, IN_T, IN_D = motion_sample.shape
OUT_T, OUT_D = IN_T, 80  # Output is 80 dimensions (muscle activations), not same as input
# train_loader_iter = dataset_MOT_MCS.cycle(train_loader)
# train_loader_iter = dataset_MOT_segmented.cycle(train_loader)

# Hyperparameters
input_dim = IN_T * IN_D # 196 * 33 Flattened input dimension

# Adjust hidden dimension for the new input size
hidden_dim = max(1024, IN_T * IN_D // 4)
output_dim = OUT_T * OUT_D
num_epochs = 10
# num_epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
# Instantiate the model
model = MLPModel(input_dim, hidden_dim, output_dim).to(device)
print("Model:", model)
# Example input (batch_size = batch_size, 196 time steps, 33 features)
# input_tensor = torch.randn(batch_size, IN_T, IN_D)
# output_tensor = model(input_tensor)

# print(output_tensor.shape)  # Should print torch.Size([batch_size, 196, 78])

# Loss function and optimizer
criterion = nn.MSELoss()  # Example for regression tasks
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1,weight_decay=1e-2)

best_test_loss = float('inf')
best_test_loss_epoch = -1
best_model = None
restore_cnt = 0 
currnet_lr = 1

# Training loop (with test loss calculation)
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0  # Track training loss for the epoch
    
    for batch in train_loader:
        # batch: (motion, len_motion, name, subject_id)
        inputs = batch[0]
        optimizer.zero_grad()

        # Forward pass
        inputs = inputs.float().to(device)
        outputs = model(inputs)

        # Reconstruction loss
        targets = inputs
        loss_main = criterion(outputs, targets)
        loss_temporal = criterion(outputs[:, 1:], outputs[:, :-1])
        loss = loss_main + loss_temporal
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()  # Accumulate loss for reporting

    # Calculate average training loss
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1},Train Loss: {avg_train_loss:.4f}")

    # Evaluation phase on test data
    model.eval()
    test_loss = 0.0
    test_l = 0.0
    avg_l_per_timestep = torch.zeros(OUT_T).to(device)
    avg_cum_thigh_activation = 0.0
    avg_cum_thigh_activation_pred = 0.0

    # Create a simple test loader
    test_loader = retargeted183_data_loader(window_size=window_size,
                                            unit_length=4,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            data_dir='/home/mnt/datasets/183_retargeted',
                                            pre_load=False)

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
            # optional per-channel metrics if applicable
            if OUT_D > 4:
                l = criterion(outputs[:, :, :OUT_D-4], inputs[:, :, :OUT_D-4])
                test_l += l.item()
                avg_cum_thigh_activation += torch.sum(inputs[:, :, :OUT_D-4]) / 4
                avg_cum_thigh_activation_pred += torch.sum(outputs[:, :, :OUT_D-4]) / 4
                avg_l_per_timestep += torch.sum((outputs[:, :, :OUT_D-4] - inputs[:, :, :OUT_D-4]) ** 2, dim=0).mean(dim=-1)



        # # Calculate the residuals

        # Solve the least squares problem
        # solution = torch.linalg.lstsq(inputs, targets)
        # residuals = solution.residuals

        # # Extract the solution (X) and residuals
        # print("Norm of lsqt:", solution.solution.norm())
        # print("Norm of NN-weight:", )    

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    avg_test_l = test_l / len(test_loader)
    avg_cum_thigh_activation = avg_cum_thigh_activation / len(test_loader)
    avg_cum_thigh_activation_pred = avg_cum_thigh_activation_pred / len(test_loader)

    avg_l_per_timestep = torch.sqrt(avg_l_per_timestep / len(test_loader))
    print(f"Epoch {epoch+1}, Best model:{best_test_loss_epoch} Test Loss: {avg_test_loss:.6f} Norm:{model.fc_main.weight.norm()} Thigh loss:{avg_test_l:.6f}")
    



    if epoch - best_test_loss_epoch > 20: # Interval between saving models
        if avg_test_loss > best_test_loss:
            model.load_state_dict(best_model)
            print("Model restored at epoch:", epoch,  " to best model at epoch:", best_test_loss_epoch)

            restore_cnt += 1
            if restore_cnt > 10:
                # Reduce learning rate to 1/10th of its value
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    print("Learning rate reduced to 1/10th of its value")

                    currnet_lr = min(currnet_lr, param_group['lr'])

                if currnet_lr < 1e-5:
                    print(f"Learning rate is less than 1e-5. Exiting:{currnet_lr}")
                    break
                restore_cnt = 0

        

        else: 

            best_model = model.state_dict()
            best_test_loss_epoch = epoch
            torch.save(model.state_dict(), 'surrogate_model.pth')
            print("Model saved at epoch:", best_test_loss_epoch, " Loss:", avg_test_loss, " Prev loss:", best_test_loss)
            best_test_loss = avg_test_loss

        print(f"Expected Thigh activation:{avg_cum_thigh_activation} Predicted:{avg_cum_thigh_activation_pred}")
        print(f"RMSE per timestep:{avg_l_per_timestep}")

# Print the weights of fc1
# fc1_weights = model.fc1.weight.data
# print("Weights of fc1 layer:")
# print(fc1_weights)

# # Check closeness to identity matrix
# identity_matrix = torch.eye(fc1_weights.size(0), fc1_weights.size(1))
# difference = fc1_weights - identity_matrix
# norm_difference = torch.norm(difference)
# print("Norm of the difference between fc1 weights and identity matrix:")
# print(norm_difference)


# Final inference: run single-sample loader and save predictions
final_test_loader = retargeted183_data_loader(window_size=window_size,
                                             unit_length=4,
                                             batch_size=1,
                                             num_workers=0,
                                             data_dir='/home/mnt/datasets/183_retargeted',
                                             pre_load=False)

collate_predictions = {}
for batch in final_test_loader:
    try:
        inputs = batch[0].float().to(device)
        names = batch[2]
        subject_ids = batch[3]
        outputs = model(inputs)

        name = names[0]
        subject_id = subject_ids[0]
        pred = outputs[0].detach().cpu().numpy()
        collate_predictions[name] = (subject_id, pred)
    except Exception as e:
        print("Error: processing final output", e)

save_dir = os.path.join('/home/mnt/datasets/183_retargeted', 'surrogate_activations')
os.makedirs(save_dir, exist_ok=True)
for name, (subject_id, arr) in collate_predictions.items():
    act_name = f"{subject_id}-{name}.mot"
    print("Saving to", os.path.join(save_dir, act_name))
    write_muscle_activations(os.path.join(save_dir, act_name), arr)