#
# Makefile for tests
#

.PHONY: all clean

all:
	@$(MAKE) -C node all
	@$(MAKE) -C simple all
	@$(MAKE) -C pytorch-operator all
	@$(MAKE) -C pytorch-converted all

clean:
	@$(MAKE) -C node clean
	@$(MAKE) -C simple clean
	@$(MAKE) -C pytorch-operator clean
	@$(MAKE) -C pytorch-converted clean
