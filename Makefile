#
# Top Makefile
#

.PHONY: all clean

all:
	@$(MAKE) -C src all
	@$(MAKE) -C examples all
	@$(MAKE) -C tests all

clean:
	@$(MAKE) -C src clean
	@$(MAKE) -C examples clean
	@$(MAKE) -C tests clean
