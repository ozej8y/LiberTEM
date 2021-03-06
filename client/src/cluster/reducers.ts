import { AllActions } from "../actions";
import { ConnectRequestParams } from "../messages";
import * as clusterActions from './actions';

export type ClusterConnectionState = {
    status: "connected",
    params: ConnectRequestParams,
} | {
    status: "disconnected",
} | {
    status: "unknown",
}

const initialClusterConnectionState: ClusterConnectionState = {
    status: "unknown"
}

export function clusterConnectionReducer(state = initialClusterConnectionState, action: AllActions): ClusterConnectionState {
    switch (action.type) {
        case clusterActions.ActionTypes.NOT_CONNECTED: {
            return {
                status: "disconnected"
            };
        }
        case clusterActions.ActionTypes.CONNECTED: {
            return {
                status: "connected",
                params: action.payload.params,
            }
        }
    }
    return state;
}