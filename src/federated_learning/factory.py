from .server import FedAvgServer, FedProtoServer, FedCMFSLServer
from .client import FedAvgClient, FedProtoClient, FedCMFSLClient
from src.models.protonet import ResNet12


class FederatedFactory:
    @staticmethod
    def get_components(args, s2_clients, s1_clients, device):
        """
        Returns (Server Instance, Client Class, Shared Body)
        """
        if args.method == "FedAvg":
            # FedAvg needs a shared body (global weights)
            shared_body = ResNet12(in_channels=10).to(device)
            server = FedAvgServer(shared_body, s2_clients, s1_clients)
            return server, FedAvgClient, shared_body

        elif args.method == "FedProto":
            # FedProto doesn't share weights, it shares prototypes
            server = FedProtoServer(s2_clients, s1_clients, lam=0.1)
            return server, FedProtoClient, None
        elif args.method == "FedCMFSL":
            server = FedCMFSLServer(
                s2_clients=s2_clients,
                s1_clients=s1_clients,
                lam1=0.1,
                lam2=0.05,
                temperature=0.1,
                bank_max_history=10,
            )
            return server, FedCMFSLClient, None

        else:
            raise ValueError(f"Unknown federated method: {args.method}")
