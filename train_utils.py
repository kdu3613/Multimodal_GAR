import torch
import time


def LiDAR_feature_processing(data_dict):
    # sharaed_feature shape : [batch_size * max_num_proposal in batch, 256]
    # return to : [batch_size, max_num_proposals (const, ex) 100), 256]
    MAX_NUM_PROPOSAL = 100
    
    shared_feature = data_dict['shared_feature']
    batch_size = data_dict['batch_size']
    # gt_bboxes = data_dict['gt_bboxes']
    
    _N, _D = shared_feature.shape
    
    res = torch.zeros([batch_size, MAX_NUM_PROPOSAL, _D])

    shared_feature = shared_feature.reshape([batch_size, -1, _D])
    
    res[:,:shared_feature.shape[1],:] = shared_feature
    
    
    return res

def sid2AdjMat(a):
    n = a.shape[0]
    b = torch.zeros((n, n))
    
    for i in range(n):
        if a[i] == -1:
            break
        for j in range(n):
            if j == i:
                b[i][j] = 1
            elif a[j] == -1:
                break
            elif a[i] == a[j]:
                b[i][j] = 1
                b[j][i] = 1
    
    for i in range(n):
        if a[i] == -1:
            for j in range(n):
                b[i][j] = -1
                b[j][i] = -1
    
    return b

def batch_sid2AdjMat(a):
    b, n = a.shape[0], a.shape[1]
    res = []
    
    for i in range(b):
        output = sid2AdjMat(a[i])
        res.append(output)
    
    return res

def Adj2Deg(A):
    """
        Adjacent matrix A -> Degree Matrix D, A : (batch_size, Max_num_proposal, Max_num_proposal)
    """
    D = torch.sum(A, dim=1)
    batch_size, Max_num_proposal = A.shape[0], A.shape[1]
    size = D.shape[0]
    res = torch.zeros(
        [batch_size, Max_num_proposal, Max_num_proposal])

    for i in range(size):
        res[i, :, :] = torch.diag(D[i, :])

    return res

def Adj2Lap(A):
    """
    Adjacent matrix A -> Laplacian matrix L (batch_size, Max_num_proposal, Max_num_proposal)
    """
    D = Adj2Deg(A)
    D = D.to(A.device)
    return D - A

def get_num_person(person_id):
    _b = person_id.shape[0]
    res = []
    for i in range(_b):
        res.append(len(torch.unique(person_id[i])) - 1)
    return res

def get_num_social_group(social_group_id):
    _b = social_group_id.shape[0]
    res = []
    for i in range(_b):
        res.append(len(torch.unique(social_group_id[i])) - 1)
    return res

def get_adjacency(social_group_id, person_num):
    _b, _ = social_group_id.shape
    res = []
    for b in range(_b):
        social_group_id_b = social_group_id[b][:person_num[b]]
        res_b = torch.zeros((person_num[b], person_num[b]))
        for i in range(person_num[b]):
            for j in range(person_num[b]):
                if i == j:
                    res_b[i][j] = 1
                elif social_group_id_b[i] == social_group_id_b[j]:
                    res_b[i][j] = 1
                    res_b[j][i] = 1
        res.append(res_b.cuda())
    return res

def get_laplacian(A):
    D = torch.diag(torch.sum(A, dim=1))
    L= D - A
    return L

def get_eig_loss2(A_theta_list, A_hat_list,alpha=1., beta=1.):
    eig_loss = torch.zeros((1,), requires_grad=True).to(A_theta_list[0].device)
    for i in range(len(A_theta_list)):
        A_hat = A_hat_list[i]
        A_theta = A_theta_list[i]
        L_theta = get_laplacian(A_theta).double()
        L_hat = get_laplacian(A_hat).double()
        
        (evals, evecs) = torch.linalg.eig(torch.matmul(L_hat.T, L_hat))
        # get eigenvectors and eigenvalues

        # get zero eigenvectors of this gt matrix
        zero_evecs = []
        for val in range(evals.shape[0]):
            # find zero eigenvalues
            if torch.abs(evals[val]).item() == 0:
                zero_evecs.append(evecs[val].double().unsqueeze(0))
        if len(zero_evecs) == 0:
            return eig_loss
        e_hat = torch.cat(zero_evecs, dim=0).to(A_theta_list[0].device)
        first_term = torch.sum(torch.matmul((torch.matmul((torch.matmul(e_hat, L_theta.T)),L_theta)),e_hat.T))                   # e.T * L_theta.T * L_theta * e
        
        L_theta_bar = torch.matmul(L_theta, (torch.eye(L_theta.shape[0]).to(A_theta_list[0].device) - torch.matmul(e_hat.T, e_hat)))
        second_term = alpha * torch.exp(-1 * beta * torch.trace(torch.matmul(L_theta_bar.T, L_theta)))
        eig_loss += first_term + second_term 
        
        
    return eig_loss
    
def get_eig_loss(A_theta_list, A_hat_list, device):
    eig_loss = torch.zeros((1,), requires_grad=True).to(device=device)
    for i in range(len(A_theta_list)):
        this_A_hat_list = A_hat_list[i]
        this_A_theta_list = A_theta_list[i]
        this_laplacian_matrix = get_laplacian(this_A_theta_list).double()
        
        # get eigenvectors and eigenvalues
        (evals, evecs) = torch.eig(this_A_hat_list, eigenvectors=True)

        # get zero eigenvectors of this gt matrix
        zero_evecs = []
        for val in range(evals.shape[0]):
            # find zero eigenvalues
            if torch.abs(evals[val][0]).item() == 0 and torch.abs(evals[0][1]).item() == 0:
                zero_evecs.append(evecs[val].double())

        if len(zero_evecs) > 0:
            for this_zero_evec in zero_evecs:
                temp = torch.mm(this_zero_evec.reshape(1, -1), this_laplacian_matrix.t())
                temp = torch.mm(temp, this_laplacian_matrix)
                this_loss_1 = torch.mm(temp, this_zero_evec.reshape(1, -1).t())
                this_loss_2 = 1 / torch.exp(torch.trace(torch.mm(this_laplacian_matrix.t(), this_laplacian_matrix)))
                this_loss = this_loss_1 + this_loss_2
                eig_loss += this_loss.reshape(1, )
        return eig_loss
    

def get_label_from_action(action, person_num):
    """ 
    action : tensor [batch_size, MAX_Num_Proposal, 27]
    person_num : [batch_size] list  
    """
    _b = action.shape[0]
    
    pose_1_list = []
    pose_2_list = []
    pose_3_list = []
    intrctn_1_list = []
    intrctn_2_list = []
    intrctn_3_list = []
    intrctn_4_list = []
    
    for i in range(_b):
        pose_1 = action[i,:person_num[i],:3]
        pose_2 = action[i,:person_num[i],3:6]
        pose_3 = action[i,:person_num[i],6:10]
        intrctn_1 = torch.zeros((person_num[i], 2)).to(action.device)
        intrctn_2 = action[i,:person_num[i],11:14]
        intrctn_3 = action[i,:person_num[i],14:20]
        intrctn_4 = action[i,:person_num[i],20:25]
        
        pose_other1, _ = torch.max(action[i,:person_num[i],3:10], dim=1)
        pose_other2, _ = torch.max(action[i,:person_num[i],6:10], dim=1)
        
        intrctn_other2,_ = torch.max(action[i,:person_num[i],14:25], dim=1)
        intrctn_other3,_ = torch.max(action[i,:person_num[i],20:25], dim=1)
        
        pose_1 = torch.cat((pose_1, pose_other1.unsqueeze(1)),dim=1)
        pose_2 = torch.cat((pose_2, pose_other2.unsqueeze(1)),dim=1)
        
        intrctn_2 = torch.cat((intrctn_2, intrctn_other2.unsqueeze(1)), dim=1)
        intrctn_3 = torch.cat((intrctn_3, intrctn_other3.unsqueeze(1)), dim=1)
        
        intrctn_1[:,0], _ = torch.max(action[i,:person_num[i],11:25], dim=1)
        intrctn_1[:,1] = 1 - intrctn_1[:,0]
        
        pose_1_list.append(pose_1)
        pose_2_list.append(pose_2)
        pose_3_list.append(pose_3)
        intrctn_1_list.append(intrctn_1)
        intrctn_2_list.append(intrctn_2)
        intrctn_3_list.append(intrctn_3)
        intrctn_4_list.append(intrctn_4)
    
    return pose_1_list, pose_2_list, pose_3_list, intrctn_1_list, intrctn_2_list, intrctn_3_list, intrctn_4_list

class Timer:
    def __init__(self):
        self.iter_old_time = time.time()
        self.epoch_old_time = time.time()
    
    def reset(self):
        self.iter_old_time = time.time()
        self.epoch_old_time = time.time()
    
    def itertime(self):
        res = time.time() - self.iter_old_time
        self.iter_old_time = time.time()
        return res    

    def epochtime(self):
        res = time.time() - self.epoch_old_time
        self.epoch_old_time = time.time()
        return res

Timer 