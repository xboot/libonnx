#
# onnx top directory
#
ONNX_DIR = $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))

#
# onnx
#
INCDIRS		+=	$(ONNX_DIR)
SRCDIRS		+=	$(ONNX_DIR) \
				$(ONNX_DIR)/default