#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import base64
import sys
import urllib2


class NsClient:

    def __init__(self):
        self.__etcd_address = 'st5-etcd-meta-1.prod.yiran.com:2379'
        (self.__server_ip, self.__server_port) = self.__find_pns_server()

    def __find_pns_server(self):
        url = 'http://%s/v3alpha/kv/range' % self.__etcd_address
        data = {
            'key': 'L3Buc19zZWFyY2hfaW5mcmFfc3Q1L3NlcnZlcl9tZXRh'
        }
        r = urllib2.urlopen(url, data=json.dumps(data))
        json_ret = json.loads(r.read())
        base64_value = json_ret["kvs"][0]["value"]
        base64_decoded = base64.b64decode(base64_value)
        server_json = json.loads(base64_decoded)

        return server_json["ip"], int(server_json["port"])

    def get_response_by_ns(self, pns):
        url = 'http://%s:%d/NSRpcService/subscribe' % (self.__server_ip, self.__server_port)
        data = {
            'subscriberId': 'pnsctl',
            'subscribeServices': [
                {
                    'serviceName': pns,
                    'currentVersion': 0
                }
            ],
            "showUnhealth": False,
            "showDisabled": False
        }
        r = urllib2.urlopen(url, data=json.dumps(data))
        return json.loads(r.read())

    def get_instances_by_ns(self, pns):
        response = self.get_response_by_ns(pns)
        server_list = []
        for instance in response["serviceInfos"][0]["instanceInfos"]:
            ip = instance["meta"]["ip"]
            port = instance["meta"]["brpcPort"]
            server_list.append((ip, port))

        return server_list

    def get_instances_with_partition_order(self, pns):
        response = self.get_response_by_ns(pns)
        instances = response["serviceInfos"][0]["instanceInfos"]
        server_list = []
        partition_num = -1
        for instance in instances:
            meta = instance["meta"]
            if partition_num == -1:
                partition_num = meta['partitionNum']
                server_list = [None] * partition_num
            ip = meta['ip']
            port = meta['brpcPort']
            server_list[int(meta.get('partitionId', 0))] = str(ip) + ':' + str(port)
        return server_list

    def get_all_instances(self, pns):
        return [str(server[0]) + ':' + str(server[1]) for server in NsClient().get_instances_by_ns(pns)]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(-1)

    pns = sys.argv[1]
    ns_client = NsClient()
    print(','.join(ns_client.get_all_instances(pns)))

