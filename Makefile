#
# Top Makefile
#

.PHONY: all clean

all:
	$(MAKE) -C examples/hello all
	$(MAKE) -C examples/benchmark all
	$(MAKE) -C examples/mnist all
	$(MAKE) -C tests all

clean:
	$(MAKE) -C examples/hello clean
	$(MAKE) -C examples/benchmark clean
	$(MAKE) -C examples/mnist clean
	$(MAKE) -C tests clean
