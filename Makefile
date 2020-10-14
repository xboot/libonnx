#
# Top Makefile
#

.PHONY: all clean

all:
	@$(MAKE) -s -C src all
	@$(MAKE) -s -C examples/hello all

clean:
	@$(MAKE) -s -C src clean
	@$(MAKE) -s -C examples/hello clean