import torch
import argparse
import matplotlib.pyplot as plt
from gflownet.env.sr_env import SRTree
from actions import Action
from policy import RNNForwardPolicy, CanonicalBackwardPolicy, TransformerForwardPolicy
from gflownet import GFlowNet, trajectory_balance_loss
from torch.autograd import forward_ad
from tqdm import tqdm


def train_plot(errs, flows, avg_mses, top_mses):
    fix, axis = plt.subplots(2, 2, figsize=(10, 10))
    ax1, ax2 = axis[0, 0], axis[0, 1]
    ax3, ax4 = axis[1, 0], axis[1, 1]

    xs = range(1, len(errs) + 1)
    ax1.plot(xs, errs, "b", label="loss over time")
    ax1.set_xlabel("20 epochs")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(xs, flows, "b", label="total flow over time")
    ax2.set_xlabel("20 epochs")
    ax2.set_ylabel("total flow")
    ax2.legend()

    ax3.plot(xs, avg_mses, "b", label="median eval mse")
    ax3.set_xlabel("20 epochs")
    ax3.set_ylabel("mse")
    ax3.legend()

    ax4.plot(xs, top_mses, "b", label="top eval mse")
    ax4.set_xlabel("20 epochs")
    ax4.set_ylabel("mse")
    ax4.legend()

    plt.show()


def train_gfn_sr(batch_size, num_epochs, show_plot=False, use_gpu=True):
    # INITIALIZE PARAMETERS AND MODEL
    # torch.manual_seed(4321) this seed doesn't work
    torch.manual_seed(42)
    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    print("training started with device", device)
    X = torch.empty(100, 2).uniform_(0, 1) * 5
    y = X[:, 0]**2 - X[:, 1]**2 + X[:, 1] - 1

    action = Action(X.shape[1])
    #env = SRTree(X, y, action_space=action, max_depth=3, loss="dynamic") #original
    env = SRTree(X, y, action_space=action, max_depth=3, loss="dynamic", verbose=False)
    # forward_policy = RNNForwardPolicy(batch_size, 250, env.num_actions, 1, model="lstm", device=device)
    forward_policy = TransformerForwardPolicy(
        num_actions=env.num_actions,
        d_model=128,
        nhead=4,
        num_layers=2,
        one_hot=True,
        device=device
    )

    
    # forward_policy = RandomForwardPolicy(env.num_actions)
    backward_policy = CanonicalBackwardPolicy(env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)

    # START TRAINING OF MODEL
    flows, errs, avg_mses, top_mses = [], [], [], []

    for i in (p := tqdm(range(num_epochs))): #for progress visualization
        s0 = env.get_initial_states(batch_size) #retrieve intial staties for the RL environment
        s, log = model.sample_states(s0) #sample states from the environment using the GFlownet model, resulting trajectory is stored in 'log'
        loss = trajectory_balance_loss(log.total_flow.to(device), #loss function is calculated based on the trajectory obtained via sampling
                                       log.rewards.to(device),
                                       log.fwd_probs.to(device))
        loss.backward() #backpropogation to compute gradients
        opt.step() #update model parameters using the optimizer's step
        opt.zero_grad() #gradients are zeroed after the optimization step

        if i % 20 == 0:
            # avg_reward = log.rewards.mean().item()
            p.set_description(f"{loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())
            avg_mse, top_mse = evaluate_model(env, model, eval_bs=100)
            avg_mses.append(avg_mse.item())
            top_mses.append(top_mse.item())

    # codes for plotting loss & rewards
    if show_plot:
        train_plot(errs, flows, avg_mses, top_mses)

    return model, env, errs, avg_mses, top_mses

def train_gfn_sr_adapted(X,y,batch_size, num_epochs, show_plot=False, use_gpu=True, policy="RNN", verbose=False):
    # INITIALIZE PARAMETERS AND MODEL
    # torch.manual_seed(4321) this seed doesn't work
    torch.manual_seed(42)
    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    print("training started with device", device)

    action = Action(X.shape[1])
    #env = SRTree(X, y, action_space=action, max_depth=3, loss="dynamic") #original
    env = SRTree(X, y, action_space=action, max_depth=3, loss="dynamic", verbose=verbose)

    if policy == "RNN":
        forward_policy = RNNForwardPolicy(256, 250, env.num_actions, 1, model="lstm", device=device)
    elif policy == "TRN":
        forward_policy = TransformerForwardPolicy(
            num_actions=env.num_actions,
            d_model=256,
            nhead=8,
            num_layers=4,
            one_hot=True,
            device=device
        )
    # forward_policy = RandomForwardPolicy(env.num_actions)
    backward_policy = CanonicalBackwardPolicy(env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)

    # START TRAINING OF MODEL
    flows, errs, avg_mses, top_mses = [], [], [], []

    for i in (p := tqdm(range(num_epochs))): #for progress visualization
        s0 = env.get_initial_states(batch_size) #retrieve intial staties for the RL environment
        s, log = model.sample_states(s0) #sample states from the environment using the GFlownet model, resulting trajectory is stored in 'log'
        loss = trajectory_balance_loss(log.total_flow.to(device), #loss function is calculated based on the trajectory obtained via sampling
                                       log.rewards.to(device),
                                       log.fwd_probs.to(device))
        loss.backward() #backpropogation to compute gradients
        opt.step() #update model parameters using the optimizer's step
        opt.zero_grad() #gradients are zeroed after the optimization step

        if i % 20 == 0:
            # avg_reward = log.rewards.mean().item()
            p.set_description(f"{loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())
            avg_mse, top_mse = evaluate_model(env, model, eval_bs=100)
            avg_mses.append(avg_mse.item())
            top_mses.append(top_mse.item())

    # codes for plotting loss & rewards
    if show_plot:
        train_plot(errs, flows, avg_mses, top_mses)

    return model, env, errs, avg_mses, top_mses



def evaluate_model(env, model, eval_bs: int = 20, top_quantile: float = 0.1):
    eval_s0 = env.get_initial_states(eval_bs)
    eval_s, _ = model.sample_states(eval_s0)
    eval_mse = env.calc_loss(eval_s)
    eval_mse = eval_mse[torch.isfinite(eval_mse)]
    avg_mse = torch.median(eval_mse)
    top_mse = torch.quantile(eval_mse, q=top_quantile)

    # 

    return avg_mse, top_mse


def run_torch_profile(prof_batch=32, prof_epochs=3, use_gpu=False):
    """
    Function for profiling the computation time of the model. See
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    for documentation and examples.
    """
    print("Starting torch profiling...")
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_training"):
            train_gfn_sr(prof_batch, prof_epochs, show_plot=False, use_gpu=use_gpu)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    model, env, errs, avg_mses, top_mses = train_gfn_sr(batch_size, num_epochs, show_plot=True, use_gpu=True)

