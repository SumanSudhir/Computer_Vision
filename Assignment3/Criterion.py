import torch

class Criterion:
    def forward(input, target):
        unnormalized_probability = torch.exp(input)
        total_sum_across_batch = unnormalized_probability.sum(1)
        total_loss = 0.0

        total_loss+= torch.log(total_sum_across_batch).sum()
        
        # for i in range(input.shape[0]):
        #     total_loss += torch.log( total_sum_across_batch[i]) - input[i][target[i]]
        #     # total_loss -= input[i][target[i]]

        total_loss -= input.gather(1,target.view(-1,1)).sum()
        
        # total_loss -= input[torch.arange(input.shape[0]),target].sum()
        return float(total_loss/input.shape[0])

    def backward(input , target):
        unnormalized_probability = torch.exp(input)
        total_sum_across_batch = unnormalized_probability.sum(1)
        p_batch_j = unnormalized_probability / total_sum_across_batch.reshape(input.shape[0],1)

        # for i in range(input.shape[0]):
            # p_batch_j[i][target[i]]-=1.0
        p_batch_j[torch.arange(input.shape[0]),target]-=1.0
        

        return p_batch_j        

