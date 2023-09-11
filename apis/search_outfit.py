import sys
sys.path += ["Hash4AllFashion"]
from search_item import search

def get_net(config, logger):
    """Get network."""
    # Get net param
    net_param = config.net_param
    logger.info(f"Initializing {utils.colour(config.net_param.name)}")
    logger.info(net_param)

    assert config.load_trained is not None

    # Dimension of latent codes
    net = FashionNet(net_param, logger, config.train_data_param.cate_selection)
    # Load model from pre-trained file
    num_devices = torch.cuda.device_count()
    map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
    logger.info(f"Loading pre-trained model from {config.load_trained}")
    state_dict = torch.load(config.load_trained, map_location=map_location)
    # load pre-trained model
    net.load_state_dict(state_dict)
    logger.info(f"Copying net to GPU-{config.gpus[0]}")
    net.cuda(device=config.gpus[0])
    net.eval()  # Unable training
    return net

if __name__ == "__main__":
    print("oke")

