import { all, call, put, take, takeEvery } from "redux-saga/effects";
import * as channelActions from '../channel/actions';
import { ConnectResponse } from "../messages";
import * as clusterActions from './actions';
import { checkClusterConnection, connectToCluster } from "./api";

function* connectSaga(action: ReturnType<typeof clusterActions.Actions.connect>) {
    const conn: ConnectResponse = yield call(connectToCluster, action.payload.params);
    yield call(putClusterStatus, conn);
}

function* putClusterStatus(conn: ConnectResponse) {
    if (conn.status === "ok") {
        yield put(clusterActions.Actions.connected(conn.connection.connection));
    } else {
        yield put(clusterActions.Actions.notConnected());
    }
}

/**
 * when the channel is connected, check if cluster is connected and update status accordingly
 */
function* trackClusterConnection() {
    while (true) {
        yield take(channelActions.ActionTypes.OPEN)
        const conn: ConnectResponse = yield call(checkClusterConnection);
        yield call(putClusterStatus, conn);
    }
}

export function* clusterConnectionSaga() {
    yield takeEvery(clusterActions.ActionTypes.CONNECT, connectSaga);
    yield all([
        trackClusterConnection(),
    ])
}