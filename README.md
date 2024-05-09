# py_yolo9_openvino_infer
* github代码地址：https://github.com/any12345com/py_yolo9_openvino_infer

### 环境依赖

| 程序         | 版本               |
| ---------- |------------------|
| python     | 3.10+            |
| 依赖库      | requirements.txt |

### 安装依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


### 介绍
* 基于python+openvino开发的yolo9模型推理服务，直接运行yolo9的onnx格式的模型文件，并对外提供图片分析的接口服务

### 启动
~~~
//启动服务
python main.py

默认端口: 9703

//接口调用地址 http://127.0.0.1:9703/algorithm

POST请求参数：
{
    'image_base64':"xxx" //分析图片的base64编码
}


~~~

### 测试

~~~
//测试yolo9的onnx格式模型或openvino格式模型
python tests.py

~~~

