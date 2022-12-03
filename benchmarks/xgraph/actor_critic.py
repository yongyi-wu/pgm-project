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

from dig.xgraph.method import SubgraphX, Actor_Critic
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from dig.xgraph.utils.compatibility import compatible_state_dict
from dig.xgraph.method.subgraphx import GnnNetsNC2valueFunc, GnnNetsGC2valueFunc, gnn_score, sparsity
IS_FRESH = False


@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")
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
        actor_critic = Actor_Critic(critic=subgraphx, actor=actor)
        x_collector = XCollector()

        train_indices, test_indices = train_test_split(test_indices, test_size=0.8)

        for i, data in enumerate(dataset[test_indices]):
            print(f'{i / len(test_indices) * 100:.1f}% of {len(test_indices)}')
            data.to(device)
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            saved_MCTSInfo_list = None
            prediction_dist = model(data)
            prediction = prediction_dist.argmax(-1).item()
            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                print(f"load example {test_indices[i]}.")

            explain_result, (related_preds, maskout_node_list) = \
                subgraphx.explain(data.x, data.edge_index,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=prediction,
                                  saved_MCTSInfo_list=saved_MCTSInfo_list,
                                  is_critic=True)
            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))

            # print(f"masked node list before: {maskout_node_list}")

            # TODO: need to use the maskout node lists as correct prediction to train the GNN model
            # and come up with a new explain_result
            num_nodes = data.x.shape[0]
            one_hot_encoding = torch.isin(torch.arange(num_nodes), torch.as_tensor(maskout_node_list)).long().to(device) # [1 if i in maskout_node_list else 0 for i in range(num_nodes)]
            explanation = actor_critic.actor_step(data.x,data.edge_index, one_hot_encoding, prediction_dist)
            new_maskout_node_list_probs = torch.sigmoid(explanation.squeeze())
            selection_threshold = 0.6
            new_maskout_node_list = torch.arange(len(new_maskout_node_list_probs))[new_maskout_node_list_probs > selection_threshold]

            value_func = GnnNetsGC2valueFunc(model, target_class=prediction)
            masked_score = gnn_score(new_maskout_node_list,
                            Data(x=data.x, edge_index=data.edge_index),
                            value_func=value_func,
                            subgraph_building_method=actor_critic.critic.subgraph_building_method)

            maskout_score = gnn_score(new_maskout_node_list,
                                    Data(x=data.x, edge_index=data.edge_index),
                                    value_func=value_func,
                                    subgraph_building_method=actor_critic.critic.subgraph_building_method)

            sparsity_score = sparsity(new_maskout_node_list, Data(x=data.x, edge_index=data.edge_index),
                                    subgraph_building_method=actor_critic.critic.subgraph_building_method)

            related_preds = {
                'masked': masked_score,
                'maskout': maskout_score,
                'origin': prediction_dist[prediction].item(),
                'sparsity': sparsity_score
            }
            print(related_preds)

            # title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
            #                  f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
            #                  f'spar: {related_preds["sparsity"]:.3f}'

            # # explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            # if isinstance(dataset, SynGraphDataset):
            #     explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.max_ex_size)
            #     edge_mask = data.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
            #                 data.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
            #     edge_mask = edge_mask.float().numpy()
            #     motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
            #     accuracy = accuracy_score(edge_mask, motif_edge_mask)
            #     roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
            #     related_preds['accuracy'] = roc_auc

            # if hasattr(dataset, 'supplement'):
            #     words = dataset.supplement['sentence_tokens'][str(test_indices[i])]
            # else:
            #     words = None

            # predict_true = 'True' if prediction == data.y.item() else "False"
            # subgraphx.visualization(explain_result,
            #                         max_nodes=config.explainers.max_ex_size,
            #                         plot_utils=plot_utils,
            #                         title_sentence=title_sentence,
            #                         vis_name=os.path.join(explanation_saving_dir,
            #                                               f'example_{test_indices[i]}_'
            #                                               f'prediction_{prediction}_'
            #                                               f'label_{data.y.item()}_'
            #                                               f'pred_{predict_true}.png'),
            #                         words=words)
            # explain_result = [explain_result]

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
        actor_critic = Actor_Critic(critic=subgraphx, actor=actor)

        train_indices, test_indices = train_test_split(node_indices, test_size=0.8)

        for i, node_idx in enumerate(node_indices):
            print(f'{i / len(node_indices) * 100:.1f}% of {len(node_indices)}')
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
            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            num_nodes = data.x.shape[0]
            one_hot_encoding = torch.isin(torch.arange(num_nodes), torch.as_tensor(maskout_node_list)).long().to(device) # [1 if i in maskout_node_list else 0 for i in range(num_nodes)]
            explanation = actor_critic.actor_step(data.x, data.edge_index, one_hot_encoding, predictions_dist, node_idx=node_idx)
            new_maskout_node_list_probs = torch.sigmoid(explanation.squeeze())
            selection_threshold = 0.6
            new_maskout_node_list = torch.arange(len(new_maskout_node_list_probs))[new_maskout_node_list_probs > selection_threshold]

            value_func = GnnNetsNC2valueFunc(
                model,
                node_idx=subgraphx.mcts_state_map.new_node_idx,
                target_class=prediction
            )
            masked_score = gnn_score(new_maskout_node_list,
                            Data(x=data.x, edge_index=data.edge_index),
                            value_func=value_func,
                            subgraph_building_method=actor_critic.critic.subgraph_building_method)

            maskout_score = gnn_score(new_maskout_node_list,
                                    Data(x=data.x, edge_index=data.edge_index),
                                    value_func=value_func,
                                    subgraph_building_method=actor_critic.critic.subgraph_building_method)

            sparsity_score = sparsity(new_maskout_node_list, Data(x=data.x, edge_index=data.edge_index),
                                    subgraph_building_method=actor_critic.critic.subgraph_building_method)

            related_preds = {
                'masked': masked_score,
                'maskout': maskout_score,
                'origin': predictions_dist[node_idx, prediction].item(),
                'sparsity': sparsity_score
            }
            print(related_preds)

            # title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
            #                  f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
            #                  f'spar: {related_preds["sparsity"]:.3f}'

            # explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            # print(f"explain result after: {explain_result}")
            # if isinstance(dataset, SynGraphDataset):
            #     explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.max_ex_size)
            #     edge_mask = edge_mask.float().numpy()
            #     motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
            #     accuracy = accuracy_score(edge_mask, motif_edge_mask)
            #     roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
            #     related_preds['accuracy'] = roc_auc
            #
            # if isinstance(dataset, SynGraphDataset):
            #     motif_edge_mask = dataset.gen_motif_edge_mask(data, node_idx=node_idx)
            #     edge_masks = [edge_mask[gc_explainer.hard_edge_mask] for edge_mask in edge_masks]
            #     roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
            #                 for edge_mask in edge_masks]
            #     for target_label, related_pred in enumerate(related_preds):
            #         related_preds[target_label]['accuracy'] = roc_aucs[target_label]

            # subgraphx.visualization(explain_result,
            #                         y=data.y,
            #                         max_nodes=config.explainers.max_ex_size,
            #                         plot_utils=plot_utils,
            #                         title_sentence=title_sentence,
            #                         vis_name=os.path.join(explanation_saving_dir,
            #                                               f'example_{node_idx}.png'))
            # explain_result = [explain_result]

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
    sys.argv.append('explainers=subgraphx')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
