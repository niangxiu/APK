run :
	shutdown -c
	@rm -f *p *png log
	@rm -rf __pycache__
	python3 -u plot.py | tee log
	shutdown 20

clean :
	@rm -f *p log
	@rm -rf __pycache__

clean2 :
	@rm -f *p *png log
	@rm -rf __pycache__
