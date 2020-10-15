#
# Top Makefile
#

.PHONY: all clean

all:
	@$(MAKE) -C src all
	@$(MAKE) -C tests all
	@$(MAKE) -C examples/hello all

clean:
	@$(MAKE) -C src clean
	@$(MAKE) -C tests clean
	@$(MAKE) -C examples/hello clean