import torch

class GINConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)
        
        return X
    


class GIN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim=None):       # hidden states directly feed in the actor/critic
        super().__init__()
        
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(GINConv(hidden_dim))
        
        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].

        # self.out_proj = torch.nn.Linear(hidden_dim*(1+n_layers), output_dim)

    def forward(self, A, X):
        X = self.in_proj(X)

        hidden_states = [X]
        
        for layer in self.convs:
            X = layer(A, X)
            hidden_states.append(X)

        # X = torch.cat(hidden_states, dim=2).sum(dim=1)        # if use features of all layers to pool graph feature
        X = X.mean(dim=1)                                        # Apply mean pooling on  features of the last layer


        # X : graph pooling feature
        # hidden_states : list of features of all nodes, each for a layer

        return X, hidden_states[-1]
    




if __name__ == '__main__':
    from GIN_jsspenv import GIN_JsspEnv
    import numpy as np

    env  = GIN_JsspEnv("ft06")
    env.seed(0)

    adj, feature, legal_actions = env.reset()
    done = False

    while not done:
        a = np.random.choice(legal_actions)
        adj, feature, reward, done, legal_actions = env.step(a)