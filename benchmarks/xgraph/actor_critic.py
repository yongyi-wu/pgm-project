import os
import torch
import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder
from benchmarks.xgraph.gnnNets import get_gnnNets, GCNNet
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from torch_geometric.data import Data

from dig.xgraph.method import SubgraphX, Actor_Critic, ExplainerBase
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from dig.xgraph.utils.compatibility import compatible_state_dict
from dig.xgraph.method.subgraphx import GnnNetsNC2valueFunc, GnnNetsGC2valueFunc, gnn_score, sparsity
IS_FRESH = False
cur_lamb = 1
cur_st = 0.5
@hydra.main(config_path="config", config_name="config")
def pipeline(config):

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}_selection_threshold_{cur_st}_lambda_{cur_lamb}.json")
    print(OmegaConf.to_yaml(config))
    recorder = Recorder(config.record_filename)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        
        test_indices = loader['test'].dataset.indices

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)

    state_dict = compatible_state_dict(torch.load(os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_"
        f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
    ))['net'])

    model.load_state_dict(state_dict)

    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          config.explainers.param.reward_method)
    check_dir(explanation_saving_dir)
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)

    if config.models.param.graph_classification:
        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)
        actor = GCNNet(
            input_dim=subgraphx.model.input_dim,
            output_dim=[1, dataset.num_classes],
            gnn_latent_dim=subgraphx.model.gnn_latent_dim,
            gnn_dropout=subgraphx.model.gnn_dropout,
            gnn_emb_normalization=subgraphx.model.gnn_emb_normalization,
            gcn_adj_normalization=subgraphx.model.gcn_adj_normalization,
            add_self_loop=subgraphx.model.add_self_loop,
            gnn_nonlinear='relu',
            readout=['identity', model.readout],
            concate=subgraphx.model.concate,
            fc_latent_dim=subgraphx.model.fc_latent_dim,
            fc_dropout=subgraphx.model.fc_dropout,
            fc_nonlinear='relu',
        ).to(device)
        actor_critic = Actor_Critic(critic=subgraphx, actor=actor, batch_size=8, lamda=cur_lamb)
        base = ExplainerBase(model)
        base.device = device
        x_collector = XCollector()

        train_indices, test_indices = train_test_split(test_indices, test_size=0.8)

        # training loop
        for i, data in enumerate(dataset[test_indices]):
            print(f'{i / len(test_indices) * 100:.1f}% of {len(test_indices)}')
            data.to(device)
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            saved_MCTSInfo_list = None
            prediction_dist = model(data)
            prediction = prediction_dist.argmax(-1).item()
            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                # print(f"load example {test_indices[i]}.")

            explain_result, (related_preds, maskout_node_list) = \
                subgraphx.explain(data.x, data.edge_index,
                                max_nodes=config.explainers.max_ex_size,
                                label=prediction,
                                saved_MCTSInfo_list=saved_MCTSInfo_list,
                                is_critic=True)
            # NOTE: USE SUBGRAPHX TO SAVE
            # torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))

            # print(f"masked node list before: {maskout_node_list}")

            # TODO: need to use the maskout node lists as correct prediction to train the GNN model
            # and come up with a new explain_result
            num_nodes = data.x.shape[0]
            one_hot_encoding = torch.isin(torch.arange(num_nodes), torch.as_tensor(maskout_node_list)).long().to(device)
            explanation = actor_critic.actor_step(data.x, data.edge_index, one_hot_encoding, data.y)
            new_maskout_node_list_probs = torch.sigmoid(explanation.squeeze())
            selection_threshold = cur_st
            new_maskout_node_list = torch.arange(len(new_maskout_node_list_probs))[new_maskout_node_list_probs > selection_threshold]

            base.__set_masks__(data.x, data.edge_index)
            row, col = data.edge_index
            node_mask = torch.isin(torch.arange(num_nodes), torch.as_tensor(new_maskout_node_list)).long()
            edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
            related_preds = base.eval_related_pred(data.x, data.edge_index, [edge_mask])[0]

            explain_result = []
            related_preds = [related_preds]
            # x_collector.collect_data(explain_result, related_preds, label=0)

        # test loop
        for i, data in enumerate(dataset[train_indices]):
            with torch.no_grad():
                print(f'{i / len(train_indices) * 100:.1f}% of {len(train_indices)}')
                data.to(device)
                data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
                saved_MCTSInfo_list = None
                prediction_dist = model(data)
                prediction = prediction_dist.argmax(-1).item()
                if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{train_indices[i]}.pt')) and not IS_FRESH:
                    saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{train_indices[i]}.pt'))
                    # print(f"load example {train_indices[i]}.")

                explain_result, (related_preds, maskout_node_list) = \
                    subgraphx.explain(data.x, data.edge_index,
                                    max_nodes=config.explainers.max_ex_size,
                                    label=prediction,
                                    saved_MCTSInfo_list=saved_MCTSInfo_list,
                                    is_critic=True)
                # NOTE: USE SUBGRAPHX TO SAVE
                # torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{train_indices[i]}.pt'))

                # print(f"masked node list before: {maskout_node_list}")

                # TODO: need to use the maskout node lists as correct prediction to train the GNN model
                # and come up with a new explain_result
                num_nodes = data.x.shape[0]
                one_hot_encoding = torch.isin(torch.arange(num_nodes), torch.as_tensor(maskout_node_list)).long().to(device)
                explanation = actor_critic.test(data.x, data.edge_index)

                new_maskout_node_list_probs = torch.sigmoid(explanation.squeeze())
                selection_threshold = 0.8
                new_maskout_node_list = torch.arange(len(new_maskout_node_list_probs))[new_maskout_node_list_probs > selection_threshold]

                base.__set_masks__(data.x, data.edge_index)
                row, col = data.edge_index
                node_mask = torch.isin(torch.arange(num_nodes), torch.as_tensor(new_maskout_node_list)).long()
                edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
                related_preds = base.eval_related_pred(data.x, data.edge_index, [edge_mask])[0]

                explain_result = []
                related_preds = [related_preds]
                x_collector.collect_data(explain_result, related_preds, label=0)

    else:
        x_collector = XCollector()
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        
        predictions_dist = model(data)
        predictions = predictions_dist.argmax(-1)

        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)
        actor = GCNNet(
            input_dim=subgraphx.model.input_dim,
            output_dim=[1, dataset.num_classes],
            gnn_latent_dim=subgraphx.model.gnn_latent_dim,
            gnn_dropout=subgraphx.model.gnn_dropout,
            gnn_emb_normalization=subgraphx.model.gnn_emb_normalization,
            gcn_adj_normalization=subgraphx.model.gcn_adj_normalization,
            add_self_loop=subgraphx.model.add_self_loop,
            gnn_nonlinear='relu',
            readout=['identity', model.readout],
            concate=subgraphx.model.concate,
            fc_latent_dim=subgraphx.model.fc_latent_dim,
            fc_dropout=subgraphx.model.fc_dropout,
            fc_nonlinear='relu',
        ).to(device)
        base = ExplainerBase(model)
        base.device = device
        actor_critic = Actor_Critic(critic=subgraphx, actor=actor, batch_size=8, lamda=cur_lamb)
        x_collector = XCollector()

        train_indices, test_indices = train_test_split(node_indices, test_size=0.8)

        # train loop
        for i, node_idx in enumerate(train_indices):
            print(f'{i / len(train_indices) * 100:.1f}% of {len(train_indices)}')
            data.to(device)
            saved_MCTSInfo_list = None
            prediction = predictions[node_idx].item()

            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir,
                                                              f'example_{node_idx}.pt'))
                print(f"load example {node_idx}.")

            explain_result, (related_preds, maskout_node_list) = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=prediction,
                                  saved_MCTSInfo_list=saved_MCTSInfo_list,
                                  is_critic=True)
            # NOTE: USE SUBGRAPHX TO SAVE
            # torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            num_nodes = data.x.shape[0]
            one_hot_encoding = torch.isin(torch.arange(num_nodes), torch.as_tensor(maskout_node_list)).long().to(device)
            explanation = actor_critic.actor_step(data.x, data.edge_index, one_hot_encoding, data.y, node_idx=node_idx)
            new_maskout_node_list_probs = torch.sigmoid(explanation.squeeze())
            selection_threshold = cur_st
            new_maskout_node_list = torch.arange(len(new_maskout_node_list_probs))[new_maskout_node_list_probs > selection_threshold]

            base.__set_masks__(data.x, data.edge_index)
            row, col = data.edge_index
            node_mask = torch.isin(torch.arange(num_nodes), torch.as_tensor(new_maskout_node_list)).long()
            edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
            related_preds = base.eval_related_pred(data.x, data.edge_index, [edge_mask])[0]

        # test loop
        for i, node_idx in enumerate(test_indices):
            with torch.no_grad():
                print(f'{i / len(test_indices) * 100:.1f}% of {len(test_indices)}')
                data.to(device)
                saved_MCTSInfo_list = None
                prediction = predictions[node_idx].item()

                if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')) and not IS_FRESH:
                    saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir,
                                                                f'example_{node_idx}.pt'))
                    print(f"load example {node_idx}.")

                explain_result, (related_preds, maskout_node_list) = \
                    subgraphx.explain(data.x, data.edge_index,
                                    node_idx=node_idx,
                                    max_nodes=config.explainers.max_ex_size,
                                    label=prediction,
                                    saved_MCTSInfo_list=saved_MCTSInfo_list,
                                    is_critic=True)
                # NOTE: USE SUBGRAPHX TO SAVE
                # torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

                num_nodes = data.x.shape[0]
                one_hot_encoding = torch.isin(torch.arange(num_nodes), torch.as_tensor(maskout_node_list)).long().to(device)
                explanation = actor_critic.test(data.x, data.edge_index)
                new_maskout_node_list_probs = torch.sigmoid(explanation.squeeze())
                selection_threshold = 0.6
                new_maskout_node_list = torch.arange(len(new_maskout_node_list_probs))[new_maskout_node_list_probs > selection_threshold]

                base.__set_masks__(data.x, data.edge_index)
                row, col = data.edge_index
                node_mask = torch.isin(torch.arange(num_nodes), torch.as_tensor(new_maskout_node_list)).long()
                edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
                related_preds = base.eval_related_pred(data.x, data.edge_index, [edge_mask])[0]

                explain_result = []
                related_preds = [related_preds]
                x_collector.collect_data(explain_result, related_preds, label=0)

    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    experiment_data = {
        'fidelity': x_collector.fidelity,
        'fidelity_inv': x_collector.fidelity_inv,
        'sparsity': x_collector.sparsity,
    }

    if x_collector.accuracy:
        print(f'Accuracy: {x_collector.accuracy}')
        experiment_data['accuracy'] = x_collector.accuracy
    if x_collector.stability:
        print(f'Stability: {x_collector.stability}')
        experiment_data['stability'] = x_collector.stability

    recorder.append(experiment_settings=['subgraphx', f"{config.explainers.max_ex_size}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    selection_thresholds = [0.5,0.6,0.7,0.8,0.9]
    lambdas = [2,5,8,10,13]
    for st in selection_thresholds:
        cur_st = st
        for lamb in lambdas:
            cur_lamb = lamb
            print(f"Current lambda {cur_lamb}")
            print(f"Current selection_threshold {cur_st}")
            sys.argv.append('explainers=subgraphx')
            sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
            sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
            sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
            sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
            pipeline()
