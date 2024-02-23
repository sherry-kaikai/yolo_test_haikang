import sophon.sail as sail
import numpy as np
import threading
import time
import os
import json
import cv2
import queue
import sys
import argparse
from multiprocessing import Process
import logging

def get_imagenames(image_path):
    file_list = os.listdir(image_path)
    imagenames = []
    for file_name in file_list:
        ext_name = os.path.splitext(file_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            imagenames.append(os.path.join(image_path,file_name))
    return imagenames

class ImgDecoderThread(object):
    def __init__(self, tpu_id, image_name_list, resize_type:sail.sail_resize_type, max_que_size:int, tpu_kernel_path:str,stress_test:bool,dete_threshold:float, nms_threshold:float):
        self.resize_type = resize_type
        self.tpu_id = tpu_id
        self.image_name_list = image_name_list
        self.resize_type = resize_type
        self.tpu_kernel_path = tpu_kernel_path
        self.stress_test = stress_test
        self.dete_threshold = dete_threshold
        self.nms_threshold = nms_threshold

        self.post_que = queue.Queue(max_que_size)
        self.image_que = queue.Queue(max_que_size)

        self.alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

    def InitProcess(self, bmodel_name, process_id):
        self.process_id = process_id
        self.engine_image_pre_process = sail.EngineImagePreProcess(bmodel_name, self.tpu_id, 0)
        self.engine_image_pre_process.InitImagePreProcess(self.resize_type, True, 5, 5)
        self.engine_image_pre_process.SetPaddingAtrr()
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        self.net_w = self.engine_image_pre_process.get_input_width()
        self.net_h = self.engine_image_pre_process.get_input_height()

        output_name = self.engine_image_pre_process.get_output_names()[0]
        self.batch_size = self.engine_image_pre_process.get_output_shape(output_name)[0]
        if len(self.engine_image_pre_process.get_output_names()) == 3:
            self.input_num = 3
            self.yolov5_post = sail.tpu_kernel_api_yolov5_detect_out(self.tpu_id, [[self.batch_size, 255, 80, 80],[self.batch_size, 255, 40, 40],[self.batch_size, 255, 20, 20]], 640, 640, self.tpu_kernel_path)
        elif len(self.engine_image_pre_process.get_output_names()) == 1:
            self.input_num = 1
            self.yolov5_post = sail.tpu_kernel_api_yolov5_out_without_decode(self.tpu_id, [self.batch_size, 25200, 85], 640, 640, self.tpu_kernel_path)
        else:
            raise ValueError("tpu-kernel post only supports dim1 or dim3 output, but get {}".format(len(self.engine_image_pre_process.get_output_names())))
        if(len(self.image_name_list)%self.batch_size != 0):
            sub_num = self.batch_size - len(self.image_name_list)%self.batch_size
            for i in range(sub_num):
                self.image_name_list.append(self.image_name_list[0])

        self.run_count = int(len(self.image_name_list)/self.batch_size)
        self.loop_count = len(self.image_name_list)
        
        thread_decoder = threading.Thread(target=self.decoder_and_pushdata, 
            args=(process_id,self.tpu_id, self.image_name_list, self.engine_image_pre_process))
        
        thread_inference = threading.Thread(target=self.Inferences_and_post_thread, 
            args=(self.run_count, self.resize_type, self.tpu_id,  self.image_que,self.dete_threshold, self.nms_threshold))
    

        thread_decoder.start()
        thread_inference.start()
 

    def decoder_and_pushdata(self, process_id, tpu_id, image_name_list, PreProcessAndInference):
        time_start = time.time()
        handle = sail.Handle(tpu_id)
        if not self.stress_test:
            for image_index, image_name in enumerate(image_name_list):
                # print(image_name)
                decoder = sail.Decoder(image_name,True,tpu_id)
                bmimg = sail.BMImage()
                ret = decoder.read(handle, bmimg)  
                while(PreProcessAndInference.PushImage(process_id,image_index, bmimg) != 0):
                    logging.info("TPUID{} Porcess[{}]:[{}/{}]PreProcessAndInference Thread Full, sleep: 10ms!".format(self.tpu_id,
                        process_id,image_index,len(image_name_list)))
                    time.sleep(0.01)
            using_time = time.time()-time_start
            logging.info("TPUID{} decoder_and_pushdata thread exit, time use: {:.2f}s,avg: {:.2f}ms".format(self.tpu_id,
                using_time,using_time/len(image_name_list)*1000))
        else:
            while True:
                time_start = time.time()
                handle = sail.Handle(tpu_id)
                if not self.stress_test:
                    for image_index, image_name in enumerate(image_name_list):
                        # print(image_name)
                        decoder = sail.Decoder(image_name,True,tpu_id)
                        bmimg = sail.BMImage()
                        ret = decoder.read(handle, bmimg)  
                        while(PreProcessAndInference.PushImage(process_id,image_index, bmimg) != 0):
                            logging.info("TPUID{} Porcess[{}]:[{}/{}]PreProcessAndInference Thread Full, sleep: 10ms!".format(self.tpu_id,
                                process_id,image_index,len(image_name_list)))
                            time.sleep(0.01)
                    using_time = time.time()-time_start
                    logging.info("TPUID{} decoder_and_pushdata thread exit, time use: {:.2f}s,avg: {:.2f}ms".format(self.tpu_id,
                        using_time,using_time/len(image_name_list)*1000))

    def Inferences_and_post_thread(self, loop_count, resize_type:sail.sail_resize_type, device_id:int, img_queue:queue.Queue, dete_threshold:float, nms_threshold:float):
        cout = 0
        handle = sail.Handle(self.tpu_id)
        bmcv = sail.Bmcv(handle)
        start_time = time.time()
        time_use = 0
        results_list = []
        for i in range(loop_count):

            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(False)
            width_list = []
            height_list= []
            for index, channel in enumerate(channel_list):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())

            img_list = []
            for index, channel in enumerate(channel_list):
                img_list.append(ost_images[index])

            ocv_images, output_tensor_map, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = img_list,output_tensor_map,\
                channel_list,imageidx_list,width_list, \
                            height_list,padding_atrr
            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            if self.input_num == 3:
                result_list = self.yolov5_post.process(output_tensor_map, dete_threshold, nms_threshold)
            elif self.input_num == 1:
                result_list = self.yolov5_post.process(output_tensor_map[0], dete_threshold, nms_threshold)
            cout += 1

            results = []
            for i in range(len(result_list)):
                if len(result_list[i]) > 0:
                    results.append(np.array(result_list[i]))
                else:
                    results.append(np.empty((0,6)))
            for det, org_w, org_h, padding_atrr in zip(results, ost_ws, ost_hs, padding_atrrs):
                ratio = [padding_atrr[2] / org_w, padding_atrr[3] / org_h]
                tx1 = padding_atrr[0]
                ty1 = padding_atrr[1]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    coords = det[:, :4]
                    coords[:, [0, 2]] -= tx1  # x padding
                    coords[:, [1, 3]] -= ty1  # y padding
                    coords[:, [0, 2]] /= ratio[0]
                    coords[:, [1, 3]] /= ratio[1]

                    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                    det[:, :4] = coords.round()
            # pipeline end 
            end_time = time.time()
            time_use += (end_time-start_time)*1000

            for idx, img_idx in enumerate(imageidxs):
                res_dict = dict()
                res_dict['image_name'] = self.image_name_list[img_idx].split('/')[-1]
                res_dict['bboxes'] = []
                det = results[idx]
                for idx in range(det.shape[0]):
                    bbox_dict = dict()
                    x1, y1, x2, y2, category_id, score = det[idx]
                    bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                    bbox_dict['category_id'] = int(category_id)
                    bbox_dict['score'] = float(round(score,5))
                    res_dict['bboxes'].append(bbox_dict)
                results_list.append(res_dict)

            # for obj in objs:
            #     bmcv.rectangle(ocv_image, obj[0], obj[1], obj[2]-obj[0], obj[3]-obj[1],(0,0,255),2)
            # image = sail.BMImage(handle,ocv_image.height(),ocv_image.width(),sail.Format.FORMAT_YUV420P,sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            # bmcv.convert_format(ocv_image,image)
            # for obj in objs:
            #     txt_d = "{}".format(obj[4])
            #     bmcv.putText(image, txt_d , obj[0], obj[1], [0,0,255], 1.4, 2)
            # bmcv.imwrite("{}_{}.jpg".format(channel,image_idx),image)
            
            start_time = time.time()

        avg_time = time_use/self.loop_count

        print("Total images: {}".format(self.loop_count))
        print("Total time use: {:.2f}ms".format(time_use))
        print("Avg time use: {:.2f}ms".format(avg_time))
        print("Process {}: {:.2f} PFS".format(self.process_id, 1000/avg_time))

        with open("process{}_results.json".format(self.process_id), 'w') as jf:
            json.dump(results_list, jf, indent=4, ensure_ascii=False)

        print("Result thread exit!")

        
def process_demo(tpu_id, image_name_list, bmodel_name, process_id, max_que_size, tpu_kernel_path, stress_test,dete_threshold,nms_threshold):
    process =  ImgDecoderThread(tpu_id, image_name_list, sail.sail_resize_type.BM_PADDING_TPU_LINEAR, max_que_size, tpu_kernel_path, stress_test,dete_threshold,nms_threshold)
    process.InitProcess(bmodel_name,process_id)

if __name__ == '__main__':
    # argparse是python用于解析命令行参数和选项的标准模块，
    # argparse模块的作用是用于解析命令行参数。
    # Example: python sample/python/yolov5_multi_3output_pic_tpu_kernel_post.py --img_dir=COCO2017/val2017 --bmodel_path=/home/ljtang/workspace/sophon-demo/sample/YOLOv5_opt/models/BM1684X/yolov5s_tpukernel_int8_4b.bmodel
    parse = argparse.ArgumentParser(description="Demo for yolov5")
    parse.add_argument('--img_dir', default='../datasets/coco/val2017_1000', type=str, help="image path directory")#文件夹所在目录
    parse.add_argument('--bmodel_path', default="../models/yolov5s_tpukernel/BM1684X/yolov5s_tpukernel_int8_4b.bmodel", type=str)
    parse.add_argument('--tpu_kernel_path', default="../models/tpu_kernel_module/libbm1684x_kernel_module.so", type=str)
    parse.add_argument('--thread_num', default=6, type=int) 
    parse.add_argument('--device_id_list', nargs='+',default=[0], help='tpu id list')  
    parse.add_argument('--draw_images', type=bool, default=False, help='draw images or not') 
    parse.add_argument('--stress_test', type=bool, default=False, help='stress test or not')
    parse.add_argument('--max_que_size', type=int,default=16, help='max_que_size')  
    parse.add_argument('--dete_threshold', type=float, default=0.001, help='dete_threshold')
    parse.add_argument('--nms_threshold', type=float, default=0.6, help='nms_threshold')
    

    # 解析参数
    opt = parse.parse_args()
    logging.basicConfig(filename= f'168X_yolo_process_video_thread_is_{opt.thread_num}_core_is{opt.device_id_list}.log',filemode='w',level=logging.DEBUG)
   
    image_path = opt.img_dir
    image_names = get_imagenames(image_path)
    # sail.set_print_flag(True)
    process_count = opt.thread_num           #进程数
    max_que_size = opt.max_que_size           #缓存的大小
    bmodel_name = opt.bmodel_path

    image_count = len(image_names)
    each_count = int(image_count/process_count)

    image_name_list = []
    for i in range(process_count-1):
        image_name_list.append(image_names[i*each_count:(i+1)*each_count])
    image_name_list.append(image_names[(process_count-1)*each_count:])

    for image_name_l in image_name_list:
        print(len(image_name_l))
        print(image_name_l[0],image_name_l[-1])

    
    process_list = []
    for tpu_id in opt.device_id_list:
        for i in range(0,process_count):
            p = Process(target=process_demo, args=(int(tpu_id), image_name_list[i], bmodel_name, i, max_que_size, opt.tpu_kernel_path,opt.stress_test,opt.dete_threshold,opt.nms_threshold))
            process_list.append(p)
    
    if opt.stress_test:
        while True:
            for p in process_list:
                p.start()

            start_time = time.time()
            for p in process_list:
                p.join()

            final_results_list = []
            for i in range(0,process_count):
                with open("process{}_results.json".format(i), 'r') as jf:
                    if i < process_count - 1:
                        final_results_list.extend(json.load(jf)[:each_count])
                    else:
                        final_results_list.extend(json.load(jf)[:image_count-(process_count-1)*each_count])
                os.system("rm -f process{}_results.json".format(i))
            assert len(final_results_list) == image_count, print(len(final_results_list))
            with open("results.json", 'w') as jf:
                json.dump(final_results_list, jf, indent=4, ensure_ascii=False)
            
            total_time = time.time() - start_time
            logging.info('TPUIDs{}  process is {},total time is {},loops for one process is {},total fps is {}'.format(opt.device_id_list,opt.thread_num,\
                total_time,(len(opt.device_id_list)*image_count)/total_time))

    else:
        for p in process_list:
            p.start()

        start_time = time.time()
        for p in process_list:
            p.join()

        final_results_list = []
        for i in range(0,process_count):
            with open("process{}_results.json".format(i), 'r') as jf:
                if i < process_count - 1:
                    final_results_list.extend(json.load(jf)[:each_count])
                else:
                    final_results_list.extend(json.load(jf)[:image_count-(process_count-1)*each_count])
            os.system("rm -f process{}_results.json".format(i))
        assert len(final_results_list) == image_count, print(len(final_results_list))
        with open("results.json", 'w') as jf:
            json.dump(final_results_list, jf, indent=4, ensure_ascii=False)
        
        total_time = time.time() - start_time
        logging.info('TPUIDs{}  process is {},total time is {},loops for one process is {},total fps is {}'.format(opt.device_id_list,opt.thread_num,\
            total_time,(len(opt.device_id_list)*opt.thread_num*image_count)/total_time))