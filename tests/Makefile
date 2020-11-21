#
# Makefile for tests
#

.PHONY: all clean

all:
	@$(MAKE) -C model all
	@$(MAKE) -C node all
	@$(MAKE) -C simple all
	@$(MAKE) -C pytorch-converted all	
	@$(MAKE) -C pytorch-operator all

clean:
	@$(MAKE) -C model clean
	@$(MAKE) -C node clean
	@$(MAKE) -C pytorch-converted clean
	@$(MAKE) -C pytorch-operator clean
	@$(MAKE) -C simple clean
